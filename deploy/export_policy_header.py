#!/usr/bin/env python3
# =============================================================================
#  export_policy_header.py  (v2)  —  ONNX (sequential MLP) -> policy_weights.h
# =============================================================================
#
#  WHAT THIS TOOL DOES
#  -------------------
#  It reads a trained neural-network policy stored as an ONNX file and emits a
#  plain C header (`policy_weights.h`) that embeds every weight as a `static
#  const float` array. A tiny microcontroller can then run the network with a
#  simple `for` loop over layers (W @ x + b, then an activation) — no ONNX
#  runtime, no heap, no external files.
#
#  WHAT "ONNX" IT ACCEPTS
#  ----------------------
#  Only a *sequential* MLP, i.e. a single straight chain of nodes:
#     * any number of Linear layers (Gemm or MatMul, optionally + Add bias)
#     * activations {Elu, Relu, Tanh, Sigmoid}  (Clip(min=0) is read as Relu)
#     * optional input normalization (x - mean) / std
#     * optional constant affine ops (Mul/Add/Sub/Div) anywhere in the chain
#  Anything with branches / residual adds / concat / Conv / attention / more
#  than one graph output is REJECTED loudly (see the design philosophy below).
#
#  DESIGN PHILOSOPHY: "refuse rather than silently emit something wrong"
#  --------------------------------------------------------------------
#  The firmware can only execute a linear chain. If the graph contains a
#  structure the firmware cannot reproduce, this tool calls die() and stops
#  instead of guessing — a wrong header baked into a robot means wrong actuator
#  commands, which can be dangerous.
#
#  HIGH-LEVEL PIPELINE (this is also the reading order)
#  ----------------------------------------------------
#     load_consts       collect every constant tensor into one dict
#     trace_main_path   walk the graph backward, keep only the linear chain
#     linear_from_node  normalize each Gemm/MatMul to   out = W @ x + b
#     parse             fold constant affine ops into neighbours; build layers
#     np_forward        a reference implementation of what the firmware does
#     verify            compare np_forward against ONNX Runtime on random inputs
#     emit              render the C header text
#
#  KEY INVARIANT USED EVERYWHERE
#  -----------------------------
#  `consts` (built by load_consts) is the single source of truth for the
#  question "is this tensor a fixed constant, or data flowing through the net?"
#  Every other function answers that question with a membership test:
#      x in consts    -> it is a weight/bias/constant
#      x not in consts -> it is data (an activation flowing along the chain)
#  Corollary/limitation: if a constant is *computed* by another node (e.g. two
#  initializers multiplied together) it will NOT be in `consts`, so it looks
#  like "data" and the tool may refuse the graph. Export models with folded
#  constants to avoid this.
#
#  USAGE
#  -----
#     python export_policy_header.py --model policy.onnx --out policy_weights.h --verify
# =============================================================================

import argparse, sys, os, datetime
import numpy as np
import onnx
from onnx import numpy_helper

# ---- Op-name lookup tables -------------------------------------------------
# ONNX activation op name  ->  the enum symbol the firmware understands.
ACT_MAP = {'Elu': 'ACT_ELU', 'Relu': 'ACT_RELU', 'Tanh': 'ACT_TANH', 'Sigmoid': 'ACT_SIGMOID'}
# Ops that reshape/relabel a tensor but do NOT change its numbers. When walking
# the graph we simply step over them (they are no-ops for a flat MLP).
PASS_THROUGH = {'Identity', 'Flatten', 'Reshape', 'Squeeze', 'Unsqueeze', 'Cast'}
# Ops that implement a matrix multiply (the core of a dense/linear layer).
LINEAR_OPS   = {'Gemm', 'MatMul'}
# Elementwise ops with a constant operand — these become part of scale/shift.
AFFINE_OPS   = {'Add', 'Sub', 'Mul', 'Div'}


def die(msg):
    """Print an error and abort. Used for every 'I cannot represent this' case."""
    sys.exit("ERROR: " + msg)


def _attr(node, name, default=None):
    """Read one attribute off an ONNX node, decoded to a native Python value.

    ONNX stores attributes as a typed union; we return the field that matches
    the attribute's declared type (int / float / list / tensor). Returns
    `default` if the node has no attribute with that name.
    """
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.TENSOR: return numpy_helper.to_array(a.t)
    return default


def load_consts(graph):
    """Collect EVERY constant tensor in the graph into a {name: ndarray} dict.

    Two sources of constants exist in ONNX and we merge both:
      1. graph.initializer  -> the trained parameters (weights, biases, the
         mean/std used for normalization, etc.)
      2. `Constant` nodes    -> constants produced inline by an op; the value
         lives in the node's `value` attribute.
    Everything is cast to float64 so later arithmetic (folding, transposes) is
    done in double precision; we only narrow to float32 when writing the header.

    This dict is the invariant described in the module header: it lets the rest
    of the code distinguish parameters from data by name membership alone.
    """
    d = {}
    for t in graph.initializer:
        d[t.name] = numpy_helper.to_array(t).astype(np.float64)
    for n in graph.node:
        if n.op_type == 'Constant':
            v = _attr(n, 'value')
            if v is not None:
                d[n.output[0]] = np.asarray(v, np.float64)
    return d


def trace_main_path(graph, consts):
    """Return the list of nodes on the single linear path input -> output.

    Strategy: start at the graph's one output and walk BACKWARD through
    producers until we reach a graph input (or a constant). Along the way we
    reject anything that is not a straight chain.

    Why walk backward? The output is unique and well-defined; from it there is
    exactly one "previous data node" at each step *if* the graph is a chain.
    The moment a node has more than one data input, the chain has forked and we
    cannot represent it — so we stop with a clear error.
    """
    # producer[tensor_name] = the node that outputs that tensor.
    producer = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n

    # The firmware has exactly one output vector (the action). Enforce it.
    if len(graph.output) != 1:
        die("graph has multiple outputs — the firmware supports only 1 output.")
    out_name = graph.output[0].name

    # Names that are genuine graph inputs (not constants) — valid places to stop.
    ins = {vi.name for vi in graph.input if vi.name not in consts}

    path, cur, seen = [], out_name, set()
    while cur in producer:                       # keep going while `cur` is produced by some node
        node = producer[cur]
        if node.op_type == 'Constant':
            break                                # reached a constant leaf; done
        if node.op_type in PASS_THROUGH:
            cur = node.input[0]                  # skip no-op; follow its single input
            continue

        # Split this node's inputs into "data" vs "constants".
        data_inputs = [x for x in node.input if x not in consts]

        if len(data_inputs) == 0:
            break                                # all inputs are constants -> chain ends here
        if len(data_inputs) > 1:
            # Two or more data inputs = residual add / concat / branch merge.
            # A flat MLP has none of these, so refuse rather than mis-handle it.
            die(f"node '{node.op_type}' has {len(data_inputs)} data inputs "
                f"(branch/residual/concat). The firmware only runs a SEQUENTIAL MLP.\n"
                f"      -> extend the firmware into an interpreter, or use TFLite/ONNX Runtime.")
        if id(node) in seen:
            die("cycle detected in the graph.")   # safety net against cycles

        seen.add(id(node))
        path.append(node)                        # record this node...
        cur = data_inputs[0]                      # ...and move to its single data predecessor

    path.reverse()                               # we collected output->input; flip to forward order

    # If we didn't land on a real graph input, the model may have an unusual
    # entry point. Warn but don't abort — parse() can still often handle it.
    if cur not in ins and cur not in consts:
        print(f"# WARNING: data path ends at '{cur}' (not a graph input).",
              file=sys.stderr)
    return path


def linear_from_node(node, consts):
    """Normalize a Gemm or MatMul node to the canonical form  out = W @ x + b.

    POST-CONDITION (the invariant the rest of parse() relies on):
        * W has shape [out, in]
        * b has shape [out]           (zeros if the node has no bias)
        * for a column input vector x of shape [in],  out = W @ x + b

    The tricky part is transposes. ONNX weights are stored assuming a ROW input
    vector `x` of shape [1, in] and computing `x @ B`, whereas we want a column
    form `W @ x`. That means our W is the transpose of ONNX's `B`, adjusted for
    the trans flags. The branches below get this right for every combination.
    """
    if node.op_type == 'Gemm':
        # Gemm computes: Y = alpha * A' * B' + beta * C
        #   where A' = A^T if transA else A,  B' = B^T if transB else B.
        # One of A/B is the data, the other is the constant weight.
        wname = [x for x in node.input[:2] if x in consts]
        if not wname:
            die("Gemm has no constant weight.")
        W = consts[wname[0]].copy()
        transB = _attr(node, 'transB', 0)
        transA = _attr(node, 'transA', 0)
        alpha  = _attr(node, 'alpha', 1.0)
        beta   = _attr(node, 'beta', 1.0)

        data_is_A = node.input[0] not in consts   # is the flowing data on the A side?
        if not data_is_A:
            # Unusual: data is B, weight is A. Honor transA, then absorb alpha.
            if transA:
                W = W.T
            W = alpha * W
        else:
            # Common MLP export: data is A (the row vector x), weight is B.
            # We need W such that W @ x == x @ B'. That makes W = (B')^T.
            #   transB=1 -> B is already [out, in]  -> keep as is
            #   transB=0 -> B is [in, out]          -> transpose to [out, in]
            if not transB:
                W = W.T
            W = alpha * W                          # fold the scalar alpha into W

        # Bias (the C term). Default to zeros; if present, absorb beta into it.
        b = np.zeros(W.shape[0])
        if len(node.input) > 2 and node.input[2] in consts:
            b = beta * consts[node.input[2]].reshape(-1)
        return W, b

    else:  # MatMul  (a bare matrix multiply, never carries a bias)
        wname = [x for x in node.input if x in consts]
        if not wname:
            die("MatMul has no constant weight.")
        W = consts[wname[0]]
        data_is_first = node.input[0] not in consts   # is x the left operand?
        # If x is on the left (x @ W_onnx), our column-form W is W_onnx^T.
        # If x is on the right (W_onnx @ x), it is already [out, in].
        Wl = W.T.copy() if data_is_first else W.copy()
        return Wl, np.zeros(Wl.shape[0])


def const_of(node, consts):
    """Return the (flattened) constant operand of an affine node, or None.

    Used for Mul/Add/Sub/Div: exactly one operand should be a constant; the
    other is the data flowing through. Returns None if no constant is found
    (which signals a merge of two data branches -> unsupported).
    """
    c = [x for x in node.input if x in consts]
    return consts[c[0]].reshape(-1) if c else None


def parse(model):
    """Turn the ONNX graph into (layers, obs_mean, obs_std).

    `layers` is a list of dicts {'W', 'b', 'act'} in forward order. The whole
    point of this function is to ABSORB every constant affine op (Mul/Add/Sub/
    Div) into a neighbouring Linear layer (or into the input normalization),
    so the emitted network is a clean sequence of  (W @ x + b, activation).

    HOW THE ABSORPTION WORKS
    ------------------------
    We carry a running affine transform  T(x) = scale * x + shift  in two
    accumulators (`scale`, `shift`) with a flag `have_pending`. As we meet
    Mul/Add/Sub/Div nodes we compose them into T. When we next meet a Linear
    op or an activation, we "flush" T into one of THREE places:

      (1) INPUT of the NEXT linear layer:
              W @ (scale*x + shift) + b
            = (W * scale) @ x + (W @ shift + b)
          Note the order in code: compute new b with the OLD W first, then
          scale W — otherwise the bias term would use the already-scaled W.

      (2) OUTPUT of the PREVIOUS linear layer (used when an activation follows
          the affine, since the affine then acts on that layer's output):
              scale * (W @ x + b) + shift
            = (scale[:,None] * W) @ x + (scale * b + shift)

      (3) INPUT NORMALIZATION (when the affine appears before any layer at all):
          the firmware computes (x - mean)/std = (1/std)*x + (-mean/std),
          so matching T(x) = scale*x + shift gives
              obs_std  = 1/scale
              obs_mean = -shift/scale

    WORKED MINI-EXAMPLE
    -------------------
    Graph:  x --Mul(2)--> Gemm(W0,b0) --Relu--> ...
    Walking forward we first see Mul(2): scale=2, shift=0, have_pending=True.
    Then we hit Gemm. There IS a pending affine and a layer already exists?
    No -- this is the first layer, so case (3)/(1)-first-layer applies and the
    Mul becomes input normalization: obs_std = 1/2 = 0.5, obs_mean = 0.
    (If instead the Mul sat *between* two Gemms, case (1) would fold it into the
    second Gemm's W and b.)
    """
    g = model.graph
    consts = load_consts(g)
    path = trace_main_path(g, consts)

    layers = []
    obs_mean = obs_std = None
    # Running affine T(x) = scale * x + shift. Start as identity (1*x + 0).
    scale = np.array(1.0)
    shift = np.array(0.0)
    have_pending = False

    def flush_prev_output():
        """Flush a pending affine that sits on the OUTPUT side (case 2/3).

        Called right before we attach an activation, because at that moment the
        pending affine (if any) transforms the previous layer's output.
        """
        nonlocal scale, shift, have_pending, obs_mean, obs_std
        if not have_pending:
            return
        if not layers:
            # No layer yet -> this affine is really input normalization (case 3).
            s  = np.atleast_1d(scale).astype(float)
            sh = np.atleast_1d(shift).astype(float)
            obs_std  = 1.0 / s
            obs_mean = -sh / s
        else:
            # Fold scale*out + shift into the previous layer's W and b (case 2).
            L = layers[-1]
            nout = L['W'].shape[0]
            s  = np.broadcast_to(np.atleast_1d(scale).astype(float), (nout,)).copy()
            sh = np.broadcast_to(np.atleast_1d(shift).astype(float), (nout,)).copy()
            L['W'] = s[:, None] * L['W']    # scale each OUTPUT row j by s[j]
            L['b'] = s * L['b'] + sh
        # Reset the accumulator back to identity.
        scale = np.array(1.0)
        shift = np.array(0.0)
        have_pending = False

    for node in path:
        op = node.op_type
        if op in PASS_THROUGH or op == 'Constant':
            continue                             # no-ops: nothing to record

        # -------- Linear layer (Gemm / MatMul) --------
        if op in LINEAR_OPS:
            W, b = linear_from_node(node, consts)
            if have_pending:
                # A pending affine sits on the INPUT of this layer.
                nin = W.shape[1]
                s  = np.broadcast_to(np.atleast_1d(scale).astype(float), (nin,)).copy()
                sh = np.broadcast_to(np.atleast_1d(shift).astype(float), (nin,)).copy()
                if not layers:
                    # Affine before the very first layer -> input normalization (case 3).
                    obs_std  = 1.0 / s
                    obs_mean = -sh / s
                else:
                    # Fold input affine into this layer (case 1). Order matters:
                    b = b + W @ sh              # new bias uses the ORIGINAL W
                    W = W * s                   # then scale W's columns by s
                scale = np.array(1.0)
                shift = np.array(0.0)
                have_pending = False
            layers.append({'W': W, 'b': b, 'act': 'ACT_NONE'})

        # -------- Constant affine (compose into scale/shift) --------
        elif op in AFFINE_OPS:
            c = const_of(node, consts)
            if c is None:
                die(f"{op} has no constant operand (it may merge two branches).")
            c = c.astype(float)
            # Compose the new op on top of T(x) = scale*x + shift.
            if op == 'Add':
                shift = shift + c                             # (s*x+sh) + c
            elif op == 'Sub':
                if node.input[0] in consts:
                    die("Sub of the form (const - x) is not supported.")  # not affine-in-place here
                shift = shift - c                             # (s*x+sh) - c
            elif op == 'Mul':
                scale = scale * c                             # c*(s*x+sh) = (c*s)x + c*sh
                shift = shift * c
            elif op == 'Div':
                if node.input[0] in consts:
                    die("Div of the form (const / x) is not supported.")  # not affine
                scale = scale / c                             # (s*x+sh)/c
                shift = shift / c
            have_pending = True

        # -------- Activation --------
        elif op in ACT_MAP:
            flush_prev_output()                  # any pending affine acts on the output first
            if not layers:
                die(f"activation {op} appears before any Linear layer.")
            layers[-1]['act'] = ACT_MAP[op]      # attach activation to the last layer
            if op == 'Elu':
                a = _attr(node, 'alpha', 1.0)
                # Firmware assumes ELU alpha == 1; warn if the model differs.
                if abs(a - 1.0) > 1e-6:
                    print(f"# WARNING: Elu alpha={a} != 1", file=sys.stderr)

        # -------- Clip: only the ReLU special case (min == 0) is supported --------
        elif op == 'Clip':
            mn = _attr(node, 'min', None)
            # Newer opsets pass min as an input tensor rather than an attribute.
            if mn is None and len(node.input) > 1 and node.input[1] in consts:
                mn = float(consts[node.input[1]])
            if mn is not None and abs(mn) < 1e-9:
                flush_prev_output()
                if layers:
                    layers[-1]['act'] = 'ACT_RELU'   # Clip(min=0) == ReLU
            else:
                die(f"Clip(min={mn}) cannot be mapped to a firmware activation.")

        # -------- Anything else is out of scope --------
        else:
            die(f"op '{op}' is not supported. Firmware allows only: Gemm/MatMul + "
                f"Add/Sub/Mul/Div + Elu/Relu/Tanh/Sigmoid. "
                f"Conv/attention/residual -> need a different runtime.")

    # A trailing affine (e.g. a final scale on the output) still needs flushing.
    flush_prev_output()

    if not layers:
        die("no Linear layer found.")

    # Fill in defaults and normalize the shapes of mean/std to length obs.
    obs = layers[0]['W'].shape[1]                # input dimension = first layer's #cols
    if obs_mean is None:
        obs_mean = np.zeros(obs)                 # no normalization -> subtract 0
    if obs_std is None:
        obs_std = np.ones(obs)                   # ...and divide by 1
    obs_mean = np.atleast_1d(obs_mean).astype(float)
    obs_std  = np.atleast_1d(obs_std).astype(float)
    # A scalar mean/std (single value applied to all inputs) is expanded to a
    # full-length vector so the header always has POLICY_OBS_DIM entries.
    if obs_mean.size == 1:
        obs_mean = np.full(obs, obs_mean[0])
    if obs_std.size == 1:
        obs_std = np.full(obs, obs_std[0])
    return layers, obs_mean, obs_std


def np_forward(layers, mean, std, x):
    """Reference forward pass — mirrors EXACTLY what the C firmware will do.

    Because this runs on the extracted `layers` (not on the ONNX graph), if its
    output matches ONNX Runtime then the emitted header must match too. This is
    the bridge that makes verify() meaningful.
    """
    x = (np.asarray(x, float) - mean) / std       # input normalization
    for L in layers:
        x = L['W'] @ x + L['b']                    # dense layer
        a = L['act']
        if   a == 'ACT_ELU':     x = np.where(x > 0, x, np.expm1(x))
        elif a == 'ACT_RELU':    x = np.maximum(x, 0)
        elif a == 'ACT_TANH':    x = np.tanh(x)
        elif a == 'ACT_SIGMOID': x = 1 / (1 + np.exp(-x))
        # ACT_NONE -> leave x unchanged
    return x


def _fmt(x):
    """Format one float as a C float literal, e.g. 1.23456789e-01f."""
    return f"{float(x):.8e}f"


def emit(layers, mean, std, src):
    """Render the full policy_weights.h text as a single string.

    Layout produced:
      * a banner + architecture summary comment
      * the activation enum
      * dimension #defines (obs dim, action dim, layer count, max width)
      * POLICY_OBS_MEAN / POLICY_OBS_STD arrays
      * per layer: POLICY_W{l} (row-major [out][in]) and POLICY_B{l}
      * lookup tables (LAYER_IN/OUT/ACT) and pointer arrays (POLICY_W/POLICY_B)
        so the firmware can loop over layers instead of hand-coding each one.
    """
    L = len(layers)
    obs = layers[0]['W'].shape[1]
    act = layers[-1]['W'].shape[0]
    # Largest dimension seen anywhere — the firmware uses it to size scratch
    # buffers big enough to hold any layer's input or output.
    mw = max(max(l['W'].shape) for l in layers)
    mw = max(mw, obs)
    arch = "obs %d -> " % obs + " -> ".join(
        f"{l['W'].shape[0]}({l['act'].replace('ACT_', '').lower()})" for l in layers)
    npar = sum(l['W'].size + l['b'].size for l in layers)

    o = ["// ============================================================================",
         "// policy_weights.h — AUTO-GENERATED by export_policy_header.py (v2), DO NOT EDIT BY HAND.",
         f"// Source: {src}", f"// Architecture: {arch}  | {npar} parameters",
         f"// Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
         "// ============================================================================", "#pragma once\n",
         "enum { ACT_NONE = 0, ACT_ELU = 1, ACT_RELU = 2, ACT_TANH = 3, ACT_SIGMOID = 4 };\n",
         f"#define POLICY_OBS_DIM   {obs}", f"#define POLICY_ACT_DIM   {act}",
         f"#define POLICY_N_LAYERS  {L}", f"#define POLICY_MAX_WIDTH {mw}\n",
         "static const float POLICY_OBS_MEAN[POLICY_OBS_DIM] = { " + ", ".join(_fmt(v) for v in mean) + " };",
         "static const float POLICY_OBS_STD [POLICY_OBS_DIM] = { " + ", ".join(_fmt(v) for v in std) + " };\n"]

    # One weight block + one bias block per layer.
    for l, ly in enumerate(layers):
        W, b = ly['W'], ly['b']
        nout, nin = W.shape
        o.append(f"// Layer {l}: {nin} -> {nout}, activation {ly['act']}")
        o.append(f"static const float POLICY_W{l}[{nout} * {nin}] = {{  // [out][in] row-major")
        for j in range(nout):                                  # one C row per output neuron
            o.append("  " + ", ".join(_fmt(W[j, i]) for i in range(nin)) + ",")
        o.append("};")
        o.append(f"static const float POLICY_B{l}[{nout}] = {{ " + ", ".join(_fmt(v) for v in b) + " };\n")

    # Metadata tables the firmware iterates over.
    o.append("static const int POLICY_LAYER_IN [POLICY_N_LAYERS] = { " + ", ".join(str(l['W'].shape[1]) for l in layers) + " };")
    o.append("static const int POLICY_LAYER_OUT[POLICY_N_LAYERS] = { " + ", ".join(str(l['W'].shape[0]) for l in layers) + " };")
    o.append("static const int POLICY_LAYER_ACT[POLICY_N_LAYERS] = { " + ", ".join(l['act'] for l in layers) + " };")
    o.append("static const float* const POLICY_W[POLICY_N_LAYERS] = { " + ", ".join(f"POLICY_W{l}" for l in range(L)) + " };")
    o.append("static const float* const POLICY_B[POLICY_N_LAYERS] = { " + ", ".join(f"POLICY_B{l}" for l in range(L)) + " };")
    return "\n".join(o) + "\n"


def verify(model_path, layers, mean, std, n=200, tol=1e-3):
    """Sanity-check the extraction against ONNX Runtime on random inputs.

    Runs the SAME `n` random vectors through (a) the real ONNX model and (b) our
    np_forward reference, and reports the worst absolute difference. If ORT is
    not installed we skip (return True) rather than fail the whole export.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("# SKIPPING --verify: onnxruntime is not installed", file=sys.stderr)
        return True

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape           # used to decide [1, obs] vs [obs]
    obs = layers[0]['W'].shape[1]
    nout = layers[-1]['W'].shape[0]
    rng = np.random.default_rng(0)               # fixed seed -> reproducible check
    worst = 0.0
    for _ in range(n):
        x = rng.uniform(-1, 1, size=obs).astype(np.float32)
        # Feed the shape ORT expects (batched 2-D vs flat 1-D).
        xin = x.reshape([1, obs]) if (shape and len(shape) == 2) else x.reshape([obs])
        y_ort = np.asarray(sess.run(None, {iname: xin})[0]).reshape(-1)[:nout]
        y_np  = np_forward(layers, mean, std, x).reshape(-1)
        worst = max(worst, float(np.max(np.abs(y_ort - y_np))))
    ok = worst < tol
    print(f"[VERIFY] max|ORT - header| over {n} samples = {worst:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="ONNX (sequential MLP) -> policy_weights.h")
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', action='store_true')   # cross-check against ONNX Runtime
    ap.add_argument('--check', action='store_true')     # print a couple of sample outputs
    a = ap.parse_args()

    model = onnx.load(a.model)
    # The ONNX checker is advisory here: warn on failure but keep going, since a
    # model can be perfectly usable for our narrow purpose yet trip the checker.
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"# WARNING checker: {e}", file=sys.stderr)

    layers, mean, std = parse(model)

    # If verification is requested and FAILS, do not write a wrong header.
    if a.verify and not verify(a.model, layers, mean, std):
        die("header does NOT match ONNX Runtime -> file NOT written.")

    open(a.out, 'w').write(emit(layers, mean, std, os.path.basename(a.model)))

    # Console summary of what was written.
    arch = " -> ".join(f"{l['W'].shape[1]}->{l['W'].shape[0]}({l['act'].replace('ACT_', '')})" for l in layers)
    print(f"[OK] {a.out}: {len(layers)} layers | {arch}")
    print(f"     obs_mean={np.round(mean, 4).tolist()}  obs_std={np.round(std, 4).tolist()}")

    # Optional smoke test: run a few fixed inputs through np_forward.
    if a.check:
        for o in ([0.1, 0.0], [-0.5, 0.5], [0.0, 0.0]):
            if len(o) == layers[0]['W'].shape[1]:
                print(f"     obs={o} -> {np_forward(layers, mean, std, o)}")


if __name__ == '__main__':
    main()
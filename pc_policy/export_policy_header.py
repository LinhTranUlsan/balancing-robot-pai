"""
export_policy_header.py — Read a policy .onnx (MLP) -> export the weights as a C header
                          to EMBED DIRECTLY into ESP32 firmware (no PC needed at runtime).

  Supports: Gemm / MatMul+Add chains with Elu/Relu/Tanh/Sigmoid activations,
            and the obs normalization block (Sub mean / Div std) if Isaac Lab embeds it in the ONNX.
  Not supported: LSTM/GRU, conv, attention -> raises a clear error.

  If you only have a .pt file (rsl_rl checkpoint): export the ONNX first with
  Isaac Lab's scripts/rsl_rl/play.py (it calls export_policy_as_onnx), then run this script.

Install:  pip install onnx numpy   (onnxruntime only needed for the --verify step; recommended)
Run:      python export_policy_header.py --model example_policy.onnx \
              --out ../firmware/rl_policy_esp32s3/policy_weights.h
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

ACT_NONE, ACT_ELU, ACT_RELU, ACT_TANH, ACT_SIGMOID = 0, 1, 2, 3, 4
ACT_BY_OP = {"Elu": ACT_ELU, "Relu": ACT_RELU, "Tanh": ACT_TANH, "Sigmoid": ACT_SIGMOID}
ACT_NAME = {ACT_NONE: "ACT_NONE", ACT_ELU: "ACT_ELU", ACT_RELU: "ACT_RELU",
            ACT_TANH: "ACT_TANH", ACT_SIGMOID: "ACT_SIGMOID"}
SKIP_OPS = {"Identity", "Cast", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Clip"}


def parse_mlp(model):
    """Walk the ONNX graph along the compute chain, return (mean, std, layers).
    layers = list of [W (out,in), b (out,), act]."""
    graph = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:                      # weights live in Constant nodes (dynamo export)
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)

    mean = std = None
    layers = []          # each element: [W, b, act]

    def const_operand(node):
        for name in node.input:
            if name in consts:
                return consts[name].astype(np.float32).ravel()
        raise SystemExit(f"[LOI] Node {node.op_type} '{node.name}' khong co operand hang so.")

    for node in graph.node:
        op = node.op_type
        if op in SKIP_OPS or op == "Constant":
            continue
        if op == "Sub" and not layers:           # obs normalization: (x - mean)
            mean = const_operand(node)
        elif op == "Div" and not layers:         # obs normalization: (...) / std
            std = const_operand(node)
        elif op == "Gemm":
            W = consts[node.input[1]].astype(np.float32)
            transB = next((a.i for a in node.attribute if a.name == "transB"), 0)
            if not transB:
                W = W.T                          # to [out, in] form
            b = (consts[node.input[2]].astype(np.float32).ravel()
                 if len(node.input) > 2 else np.zeros(W.shape[0], np.float32))
            layers.append([W, b, ACT_NONE])
        elif op == "MatMul":
            W = consts[node.input[1]].astype(np.float32).T   # x@W: stored [in,out] -> [out,in]
            layers.append([W, np.zeros(W.shape[0], np.float32), ACT_NONE])
        elif op == "Add" and layers and not layers[-1][1].any():
            layers[-1][1] = const_operand(node)  # bias of the preceding MatMul
        elif op in ACT_BY_OP:
            if not layers:
                raise SystemExit(f"[LOI] Activation {op} truoc lop Linear dau tien?")
            layers[-1][2] = ACT_BY_OP[op]
        else:
            raise SystemExit(f"[LOI] Op '{op}' chua ho tro (khong phai MLP thuan? "
                             f"LSTM/conv can cach khac). Node: {node.name}")

    if not layers:
        raise SystemExit("[LOI] Khong tim thay lop Linear nao trong model.")
    obs_dim = layers[0][0].shape[1]
    mean = mean if mean is not None else np.zeros(obs_dim, np.float32)
    std = std if std is not None else np.ones(obs_dim, np.float32)
    return mean, std, layers


def np_forward(obs, mean, std, layers):
    x = (obs.astype(np.float32) - mean) / std
    for W, b, act in layers:
        x = W @ x + b
        if act == ACT_ELU:     x = np.where(x > 0, x, np.expm1(x))
        elif act == ACT_RELU:  x = np.maximum(x, 0)
        elif act == ACT_TANH:  x = np.tanh(x)
        elif act == ACT_SIGMOID: x = 1 / (1 + np.exp(-x))
    return x


def fmt_array(arr):
    vals = ", ".join(f"{v:.8e}f" for v in np.ravel(arr))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="example_policy.onnx")
    ap.add_argument("--out", default="../firmware/rl_policy_esp32s3/policy_weights.h")
    ap.add_argument("--no-verify", action="store_true", help="Bo qua doi chieu voi onnxruntime")
    args = ap.parse_args()

    model = onnx.load(args.model)
    mean, std, layers = parse_mlp(model)
    obs_dim = layers[0][0].shape[1]
    act_dim = layers[-1][0].shape[0]
    max_width = max(max(W.shape) for W, _, _ in layers)
    n_params = sum(W.size + b.size for W, b, _ in layers)

    print(f"[EXPORT] {args.model}: obs={obs_dim} -> " +
          " -> ".join(str(W.shape[0]) for W, _, _ in layers) +
          f"  ({n_params} tham so, ~{n_params*4/1024:.1f} KB)")
    if not np.allclose(mean, 0) or not np.allclose(std, 1):
        print("[EXPORT] Model co khoi chuan hoa obs (mean/std) -> da nhung vao header.")

    # Cross-check the numpy forward against onnxruntime on random obs
    if not args.no_verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(100):
            obs = (rng.standard_normal(obs_dim) * 0.5).astype(np.float32)
            ref = np.ravel(sess.run(None, {in_name: obs.reshape(1, -1)})[0])
            got = np_forward(obs, mean, std, layers)
            worst = max(worst, float(np.max(np.abs(ref - got))))
        print(f"[VERIFY] 100 obs ngau nhien, sai lech max |onnxruntime - header| = {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit("[LOI] Sai lech qua lon — graph co nhanh chua ho tro? Kiem tra lai.")

    lines = [
        "// ============================================================================",
        f"// policy_weights.h — TU DONG SINH boi export_policy_header.py, DUNG SUA TAY.",
        f"// Nguon: {args.model}",
        f"// Kien truc: obs {obs_dim} -> " +
        " -> ".join(f"{W.shape[0]}({ACT_NAME[a][4:].lower()})" for W, _, a in layers) +
        f"  | {n_params} tham so",
        "// Tao lai: python pc_policy/export_policy_header.py --model <file.onnx> --out <file nay>",
        "// ============================================================================",
        "#pragma once",
        "",
        "enum { ACT_NONE = 0, ACT_ELU = 1, ACT_RELU = 2, ACT_TANH = 3, ACT_SIGMOID = 4 };",
        "",
        f"#define POLICY_OBS_DIM   {obs_dim}",
        f"#define POLICY_ACT_DIM   {act_dim}",
        f"#define POLICY_N_LAYERS  {len(layers)}",
        f"#define POLICY_MAX_WIDTH {max_width}",
        "",
        f"static const float POLICY_OBS_MEAN[POLICY_OBS_DIM] = {{ {fmt_array(mean)} }};",
        f"static const float POLICY_OBS_STD [POLICY_OBS_DIM] = {{ {fmt_array(std)} }};",
        "",
    ]
    for i, (W, b, act) in enumerate(layers):
        lines.append(f"// Lop {i}: {W.shape[1]} -> {W.shape[0]}, activation {ACT_NAME[act]}")
        lines.append(f"static const float POLICY_W{i}[{W.shape[0]} * {W.shape[1]}] = {{  // [out][in] row-major")
        for row in W:
            lines.append("  " + fmt_array(row) + ",")
        lines.append("};")
        lines.append(f"static const float POLICY_B{i}[{b.size}] = {{ {fmt_array(b)} }};")
        lines.append("")
    n = len(layers)
    lines += [
        f"static const int POLICY_LAYER_IN [POLICY_N_LAYERS] = {{ {', '.join(str(W.shape[1]) for W, _, _ in layers)} }};",
        f"static const int POLICY_LAYER_OUT[POLICY_N_LAYERS] = {{ {', '.join(str(W.shape[0]) for W, _, _ in layers)} }};",
        f"static const int POLICY_LAYER_ACT[POLICY_N_LAYERS] = {{ {', '.join(ACT_NAME[a] for _, _, a in layers)} }};",
        f"static const float* const POLICY_W[POLICY_N_LAYERS] = {{ {', '.join(f'POLICY_W{i}' for i in range(n))} }};",
        f"static const float* const POLICY_B[POLICY_N_LAYERS] = {{ {', '.join(f'POLICY_B{i}' for i in range(n))} }};",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[EXPORT] Da ghi {args.out}")
    print("Buoc tiep: mo firmware/rl_policy_esp32s3/rl_policy_esp32s3.ino trong Arduino IDE va nap.")


if __name__ == "__main__":
    main()

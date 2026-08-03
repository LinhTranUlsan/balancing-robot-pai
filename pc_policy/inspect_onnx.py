"""
inspect_onnx.py  —  DUMP the structure + run a trial inference for an ONNX policy
                    exported from Isaac Lab (rsl_rl: export_policy_as_onnx).

Purpose: know EXACTLY how many dimensions the input observation has, the input/output names,
the data type, and try 1 inference step. Run THIS FIRST before doing anything else.

Install:  pip install onnxruntime numpy
Run:      python inspect_onnx.py --model path/to/policy.onnx
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Duong dan file .onnx")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    print("=" * 64)
    print(f"MODEL: {args.model}")
    print("-" * 64)
    print("INPUTS:")
    for i in sess.get_inputs():
        print(f"  name={i.name!r:20s} shape={i.shape} dtype={i.type}")
    print("OUTPUTS:")
    for o in sess.get_outputs():
        print(f"  name={o.name!r:20s} shape={o.shape} dtype={o.type}")
    print("=" * 64)

    inp = sess.get_inputs()[0]
    # Infer the obs dimension: take the last positive int dim (skip batch dim/-1/None/'batch')
    obs_dim = next((d for d in reversed(inp.shape) if isinstance(d, int) and d > 0), None)
    if obs_dim is None:
        obs_dim = 4
        print(f"[WARN] Khong suy ra duoc obs_dim tu shape -> tam dung {obs_dim}. Sua theo env sim!")
    else:
        print(f"[INFO] obs_dim = {obs_dim}  (kiem tra lai khop observation_space cua env sim!)")

    # Try a few sample observations
    tests = {
        "zeros": np.zeros(obs_dim, np.float32),
        "pole_tilt_+0.1rad": np.eye(1, obs_dim, 0, dtype=np.float32).ravel() * 0.1,
        "random": np.random.default_rng(0).standard_normal(obs_dim).astype(np.float32) * 0.1,
    }
    print("-" * 64)
    for name, obs in tests.items():
        out = sess.run(None, {inp.name: obs.reshape(1, -1)})
        act = np.ravel(out[0])
        print(f"[RUN] {name:18s} obs={np.round(obs,3).tolist()} -> action={np.round(act,4).tolist()}")
    print("=" * 64)
    print("Neu action doi dau/thay doi hop ly khi obs doi -> model OK.")
    print("Buoc tiep: sua assemble_obs() trong run_policy_serial.py cho khop THU TU obs nay.")


if __name__ == "__main__":
    main()

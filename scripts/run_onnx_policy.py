"""Chạy inference 1 policy đã export ra ONNX (rsl_rl), tính đầu ra (action) từ đầu vào (observation).

Ví dụ:
    python scripts/run_onnx_policy.py \
        --policy logs/rsl_rl/twip_rsl_v2/2026-07-24_12-50-41/exported/policy.onnx \
        --obs 0.1 -0.2
"""

import argparse

import numpy as np
import onnxruntime as ort


def load_session(policy_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])


def run_policy(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    obs = obs.reshape(1, -1).astype(np.float32)
    expected_dim = input_meta.shape[-1]
    if isinstance(expected_dim, int) and obs.shape[-1] != expected_dim:
        raise ValueError(f"Observation có {obs.shape[-1]} phần tử, nhưng policy yêu cầu {expected_dim}.")

    outputs = session.run(None, {input_meta.name: obs})
    return outputs[0]


def main():
    parser = argparse.ArgumentParser(description="Tính action từ observation bằng policy ONNX đã export.")
    parser.add_argument("--policy", type=str, required=True, help="Đường dẫn tới file policy.onnx")
    parser.add_argument("--obs", type=float, nargs="+", required=True, help="Vector observation đầu vào")
    args = parser.parse_args()

    session = load_session(args.policy)
    obs = np.array(args.obs)
    actions = run_policy(session, obs)

    print(f"Observation: {obs.tolist()}")
    print(f"Action:      {actions.reshape(-1).tolist()}")


if __name__ == "__main__":
    main()

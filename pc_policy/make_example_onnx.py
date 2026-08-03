"""
make_example_onnx.py — Generate an EXAMPLE ONNX file matching the REAL setup (2-wheel cart, IMU-only, 1 motor).

  Input : obs = [pole_pos (rad), pole_vel (rad/s)]   (2 dims — from IMU, NO encoder yet)
  Output: actions = [u] in [-1,1]                    (command for 1 motor)

  Architecture: MLP 2 -> 32 -> 32 -> 1 (ELU, tanh at output) — like a real RL policy.
  The weights are QUICKLY TRAINED to imitate a stable PD controller, so this file WORKS
  (the motor reacts in the right direction) rather than being random -> tests the real PC<->ESP32 pipeline.

  Later, when you actually train on Isaac Lab, just replace this .onnx with the real one;
  the pipeline (obs 2 dims, action 1) STAYS THE SAME as long as you keep observation_space = 2.

Run: python make_example_onnx.py            (needs torch; does NOT call onnx to export)
"""
import argparse
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim=2, act_dim=1, hidden=(32, 32)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, act_dim), nn.Tanh()]   # tanh -> action bounded in [-1,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example_policy.onnx")
    ap.add_argument("--kp", type=float, default=5.0, help="PD gain theo pole_pos (rad)")
    ap.add_argument("--kd", type=float, default=0.5, help="PD gain theo pole_vel (rad/s)")
    ap.add_argument("--iters", type=int, default=1500)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = Policy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    # Train to imitate PD: target = clip(-Kp*theta - Kd*omega, -1, 1)
    print("Huan luyen bat chuoc PD...")
    for it in range(args.iters):
        theta = (torch.rand(4096, 1) * 2 - 1) * 0.5   # +-0.5 rad (~+-28 deg)
        omega = (torch.rand(4096, 1) * 2 - 1) * 4.0   # +-4 rad/s
        obs = torch.cat([theta, omega], dim=1)
        target = torch.clamp(-args.kp * theta - args.kd * omega, -1.0, 1.0)
        loss = lossf(model(obs), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 300 == 0:
            print(f"  iter {it+1}/{args.iters} loss={loss.item():.5f}")

    model.eval()
    print("Kiem tra hanh vi (nghieng + -> action nen AM = day nguoc lai):")
    with torch.no_grad():
        for th, om in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.3, 0.0), (0.0, 2.0)]:
            a = model(torch.tensor([[th, om]], dtype=torch.float32)).item()
            print(f"  obs=[{th:+.2f}, {om:+.2f}] -> action={a:+.3f}")

    dummy = torch.zeros(1, 2, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,          # use the legacy exporter (TorchScript) -> no onnxscript needed
    )
    print(f"Da xuat ONNX: {args.out}")


if __name__ == "__main__":
    main()

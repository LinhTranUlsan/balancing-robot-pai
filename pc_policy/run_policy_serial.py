"""
run_policy_serial.py  —  PC-in-the-loop loop.

  ESP32 sends :  O <pole_pos> <pole_vel> <cart_pos> <cart_vel>   (radian, rad/s ; m, m/s)
  PC does     :  build sim-matching obs -> ONNX -> action -> convert to u in [-1,1] -> send "A <u>"

Install:  pip install onnxruntime onnx numpy pyserial
Run:      python run_policy_serial.py --port COM6 --model example_policy.onnx

READY TO RUN NOW with example_policy.onnx (2-dim IMU obs) — NO code changes needed.
Only edit the 2 functions below WHEN switching to a different env/observation (e.g. add encoder -> 4 dims).
The script self-checks whether the obs dimension matches the model and errors out if they differ.
"""
import argparse
import time
import numpy as np


# ==================== ASSEMBLE OBSERVATION (READY for example_policy.onnx) =======
# Return a float32 array in the exact ORDER + UNITS as the observation at TRAIN time.
# Current setup (no encoder): obs = [pole_pos, pole_vel] (2 dims, from IMU) -> READY.
# ONLY CHANGE for a different training env, e.g. add encoder + 4-dim cartpole:
#   return np.array([raw["pole_pos"], raw["pole_vel"], raw["cart_pos"], raw["cart_vel"]], np.float32)
def assemble_obs(raw: dict) -> np.ndarray:
    return np.array([
        raw["pole_pos"],   # rad
        raw["pole_vel"],   # rad/s
    ], dtype=np.float32)


# ==================== ACTION -> MOTOR COMMAND u in [-1,1] (READY) ==============
# Policy outputs action in [-1,1] (example already passed through tanh). Clip then send directly as u.
# ESP32 converts u -> duty (sign = direction, |u| -> [MIN_DUTY..MAX_DUTY]). The robot reads it correctly.
# ONLY EDIT if the motor pushes the WRONG direction -> "return -u" (or flip 'm -1' in firmware).
def action_to_motor_u(action: np.ndarray) -> float:
    u = float(np.ravel(action)[0])
    return max(-1.0, min(1.0, u))
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Cong serial ESP32 (vd COM6 hoac /dev/ttyACM0)")
    ap.add_argument("--model", required=True, help="File policy .onnx")
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    import serial
    import onnxruntime as ort

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    obs_dim = next((d for d in reversed(sess.get_inputs()[0].shape) if isinstance(d, int) and d > 0), 4)
    print(f"[PC] Model {args.model}: input={in_name!r}, obs_dim={obs_dim}")

    # Self-check: assemble_obs must return the exact number of dims the model needs (catch errors early)
    probe = assemble_obs({"pole_pos": 0.0, "pole_vel": 0.0, "cart_pos": 0.0, "cart_vel": 0.0})
    if probe.size != obs_dim:
        raise SystemExit(f"[LOI] assemble_obs() tra {probe.size} gia tri nhung model can {obs_dim}. "
                         f"Sua assemble_obs() cho khop roi chay lai.")

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"[PC] Da mo {args.port} @ {args.baud}. Bat dau vong lap. Ctrl+C de dung.")

    n = 0
    t_report = time.time()
    lat_sum = 0.0
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):           # log/comment line from ESP32
                print("[ESP32]", line)
                continue
            if not line.startswith("O"):
                continue

            t0 = time.perf_counter()
            try:
                vals = [float(x) for x in line.split()[1:]]
            except ValueError:
                continue
            raw = {
                "pole_pos": vals[0] if len(vals) > 0 else 0.0,
                "pole_vel": vals[1] if len(vals) > 1 else 0.0,
                "cart_pos": vals[2] if len(vals) > 2 else 0.0,
                "cart_vel": vals[3] if len(vals) > 3 else 0.0,
            }
            obs = assemble_obs(raw).reshape(1, -1)
            action = sess.run(None, {in_name: obs})[0]
            u = action_to_motor_u(action)
            ser.write(f"A {u:.4f}\n".encode())

            lat_sum += time.perf_counter() - t0
            n += 1
            if time.time() - t_report >= 1.0:
                print(f"[PC] {n:3d} Hz | infer+send ~{1000*lat_sum/max(n,1):.2f} ms "
                      f"| u={u:+.3f} | obs={np.round(obs.ravel(),3).tolist()}")
                n = 0
                lat_sum = 0.0
                t_report = time.time()
    except KeyboardInterrupt:
        ser.write(b"A 0\n")   # stop the motor on exit
        ser.flush()
        print("\n[PC] Dung. Da gui lenh motor = 0.")


if __name__ == "__main__":
    main()

# balancing-robot-pai

A complete **reinforcement-learning (RL) pipeline for a two-wheeled inverted pendulum (TWIP) self-balancing robot** — trained in **NVIDIA Isaac Lab / Isaac Sim**, exported to **ONNX**, converted to a plain C header, and deployed to an **ESP32-S3** running an L298N motor driver.

> This project uses **RL (PPO)**, not a hand-tuned PID controller. The policy is a small MLP
> (`2 → 32 → 32 → 1`, ELU) that maps `[pitch, pitch_rate]` to a single symmetric wheel torque.

```
Isaac Lab (train PPO)  ──►  policy.onnx  ──►  policy_weights.h  ──►  ESP32-S3 firmware
   scripts/rsl_rl/          play.py           pc_policy/            firmware/…_l298n
```

---

## 1. What's in here

| Path | Purpose |
|------|---------|
| `source/Twip_Rsl_v2/` | The Isaac Lab extension (manager-based RL env, task `Template-Twip-Rsl-V2-v0`). **This is the core.** |
| `assets/TwoWheel.urdf` | Robot description used by the simulation. |
| `assets/TwoWheel/…*.usd` | Pre-converted USD stage of the robot (optional; the env spawns from the URDF). |
| `scripts/` | Launchers: `list_envs.py`, `zero_agent.py`, `random_agent.py`, `run_onnx_policy.py`, and `rsl_rl/{train,play,cli_args,plot_pitch}.py`. |
| `pc_policy/` | Convert / inspect ONNX policies: `export_policy_header.py`, `inspect_onnx.py`, `make_example_onnx.py`, `run_policy_serial.py`, plus a ready-to-flash demo `example_policy_strong.onnx`. |
| `firmware/rl_policy_esp32s3_l298n/` | ESP32-S3 Arduino sketch (`.ino`), the generated `policy_weights.h`, and wiring/usage notes. |
| `pyproject.toml` | Ruff / formatting / pytest config for the repo. |

### The robot (hardware this firmware targets)
- **MCU:** ESP32-S3 (Arduino core ≥ 3.0)
- **IMU:** BMI160 over I²C — `SDA = GPIO6`, `SCL = GPIO5`
- **Motor driver:** L298N — left motor `IN1=7, IN2=8, ENA=11`; right motor `IN3=9, IN4=10, ENB=12`
- **Encoders (optional):** left `A=1, B=2`; right `A=4, B=13`
- **Control rate:** **100 Hz** (must match training: sim `dt=1/200` × `decimation=2`)

---

## 2. Prerequisites

### For training / simulation (`source/`, `scripts/`)
1. **NVIDIA RTX GPU** + recent driver.
2. **Isaac Sim 5.1** (also compatible with 4.5 / 5.0) **and Isaac Lab**, installed per the
   [official Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   Python **3.10 or 3.11**.
3. **rsl-rl-lib ≥ 3.0.1** (usually ships with Isaac Lab; `train.py` prints the exact
   `pip install` command if the version is wrong).

> ⚠️ Everything under `scripts/` and `source/` must be run with the **Isaac Lab Python
> interpreter**, not a plain system `python`. Either activate the Isaac Lab conda env, or
> prefix commands with the launcher:
> - Windows: `<path-to-IsaacLab>\isaaclab.bat -p <script>`
> - Linux: `<path-to-IsaacLab>/isaaclab.sh -p <script>`

### For ONNX → C header (`pc_policy/`)
```bash
pip install onnx onnxruntime numpy
# torch is only needed if you run make_example_onnx.py
```

### For the firmware (`firmware/`)
- **Arduino IDE** (or arduino-cli) with the **esp32 board package ≥ 3.0**.
- A BMI160 I²C library (see the sketch's includes) and the WS2812/FastLED-style LED lib if you keep the status LED.

---

## 3. Install (simulation side)

```bash
git clone https://github.com/<your-user>/balancing-robot-pai.git
cd balancing-robot-pai

# Install this extension into the Isaac Lab Python env (EDITABLE install):
python -m pip install -e source/Twip_Rsl_v2

# Verify the task registered:
python scripts/list_envs.py        # should list: Template-Twip-Rsl-V2-v0
```

---

## 4. Train → get an ONNX policy

```bash
# Train (headless is fastest). Checkpoints go to logs/rsl_rl/twip_rsl_v2/<timestamp>/
python scripts/rsl_rl/train.py --task=Template-Twip-Rsl-V2-v0 --headless

# Replay a checkpoint AND export policy.onnx (rsl_rl exports on play):
python scripts/rsl_rl/play.py  --task=Template-Twip-Rsl-V2-v0 \
    --checkpoint logs/rsl_rl/twip_rsl_v2/<timestamp>/model_<N>.pt --num_envs=16
# → the exported policy lands in logs/rsl_rl/twip_rsl_v2/<timestamp>/exported/policy.onnx
```

Useful flags: `--num_envs`, `--seed -1` (random), `--max_iterations`, `--device cuda:0`,
`--video` (implies `--enable_cameras`).

Sanity-check the env without a policy:
```bash
python scripts/zero_agent.py   --task=Template-Twip-Rsl-V2-v0
python scripts/random_agent.py --task=Template-Twip-Rsl-V2-v0
```

---

## 5. Convert ONNX → firmware header

```bash
python pc_policy/export_policy_header.py \
    --model logs/rsl_rl/twip_rsl_v2/<timestamp>/exported/policy.onnx \
    --out   firmware/rl_policy_esp32s3_l298n/policy_weights.h

# Inspect any policy's layers / test a forward pass:
python pc_policy/inspect_onnx.py --model <path>.onnx
```

Want to try the pipeline **without training first?** Use the bundled demo:
```bash
python pc_policy/export_policy_header.py \
    --model pc_policy/example_policy_strong.onnx \
    --out   firmware/rl_policy_esp32s3_l298n/policy_weights.h
```

---

## 6. Flash the ESP32-S3

1. Open `firmware/rl_policy_esp32s3_l298n/rl_policy_esp32s3_l298n.ino` in the Arduino IDE.
2. Board: **ESP32S3 Dev Module**, **USB CDC On Boot: Enabled**, esp32 core **≥ 3.0**.
3. Confirm at the top of the sketch: `CONTROL_HZ = 100` and `policy_weights.h` matches your model
   (`POLICY_OBS_DIM == 2`, `POLICY_ACT_DIM == 1`).
4. Upload. See `firmware/rl_policy_esp32s3_l298n/README.md` for the serial commands
   (pitch trim `s`, motor test, etc.).

---

## 7. Troubleshooting (common errors)

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: No module named 'Twip_Rsl_v2'` | You skipped the editable install. Run `python -m pip install -e source/Twip_Rsl_v2` **in the Isaac Lab env**. |
| `FileNotFoundError: ...TwoWheel.urdf` | Keep the repo layout intact, or set env var `TWIP_URDF=/abs/path/TwoWheel.urdf`. |
| `list_envs.py` shows nothing / task missing | Import side-effect didn't run — reinstall the extension; make sure you're using the Isaac Lab interpreter. |
| rsl-rl version error on `train.py` | Install the exact version the script prints (`rsl-rl-lib>=3.0.1`). |
| Isaac Sim crashes / GPU errors | Wrong Isaac Sim version or driver — use 5.1 (or 4.5/5.0) with a supported RTX driver. |
| Arduino: `redefinition of 'setup'/'loop'` | Keep **only one** `.ino` in the sketch folder. |
| Robot balances in sim but not on hardware | Ensure `CONTROL_HZ = 100`, calibrate pitch trim (`s` command), and verify motor/encoder wiring against the pin table above. |

---

## 8. License & credits

- Built on the [Isaac Lab](https://github.com/isaac-sim/IsaacLab) external-extension template
  (BSD-3-Clause headers retained on upstream-derived files).
- TWIP RL env, firmware, and ONNX→C tooling by the project author (PAI, University of Ulsan).

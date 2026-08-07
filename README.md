# balancing-robot-pai

A complete **reinforcement-learning (RL) pipeline for a balancing robot** — trained in **NVIDIA Isaac Lab / Isaac Sim**, exported to **ONNX**, converted to a **plain C header** of MLP weights, and deployed to an **ESP32-S3** (PlatformIO) running an L298N motor driver.

> This project uses **RL (PPO)**, not a hand-tuned PID controller. The policy is a small MLP
> (`2 → 32 → 32 → 1`, ELU) that maps `[pitch, pitch_rate]` to a single symmetric wheel torque.

```
Isaac Lab (train PPO)  ──►  policy.onnx  ──►  policy_weights.h  ──►  ESP32-S3 firmware
   scripts/rsl_rl/          play.py         (C MLP weights)         deploy/ (PlatformIO)
```

---

## 1. What's in here

| Path | Purpose |
|------|---------|
| `source/Twip_Rsl_v2/` | The Isaac Lab extension (manager-based RL env, task `Template-Twip-Rsl-V2-v0`). **This is the core.** |
| `assets/TwoWheel.urdf` | Robot description used by the simulation. |
| `assets/TwoWheel/…*.usd` | Pre-converted USD stage of the robot (optional; the env spawns from the URDF). |
| `scripts/` | Launchers: `list_envs.py`, `zero_agent.py`, `random_agent.py`, `run_onnx_policy.py`, and `rsl_rl/{train,play,cli_args,plot_pitch}.py`. |
| `deploy/` | ESP32-S3 firmware — a **PlatformIO** C/C++ project. `src/main.cpp` runs the policy with a **hand-written MLP forward** (no external ML runtime), reading the exported weights in `include/policy_weights.h`. Reads the BMI160 IMU over raw I²C and drives the L298N via a pulse-density motor shaper. Also ships `export_policy_header.py` (ONNX → `policy_weights.h` converter). |
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
2. **Isaac Sim 5.1** (also compatible with 4.5 / 5.0) **and Isaac Lab** — full step-by-step pip
   install (conda + git + Isaac Sim + Isaac Lab) is in **Section 3** below. Python **3.10 or 3.11**.
3. **rsl-rl-lib ≥ 3.0.1** (usually ships with Isaac Lab; `train.py` prints the exact
   `pip install` command if the version is wrong).

> ⚠️ Everything under `scripts/` and `source/` must be run with the **Isaac Lab Python
> interpreter**, not a plain system `python`. Either activate the Isaac Lab conda env, or
> prefix commands with the launcher:
> - Windows: `<path-to-IsaacLab>\isaaclab.bat -p <script>`
> - Linux: `<path-to-IsaacLab>/isaaclab.sh -p <script>`

### For on-device deployment (`deploy/`)
- **PlatformIO** — either the VS Code **PlatformIO IDE** extension, or the CLI: `pip install platformio`.
- On first build PlatformIO installs the ESP32 toolchain (and any libraries listed in `platformio.ini`)
  automatically. `src/main.cpp` itself is self-contained — raw-I²C BMI160, interrupt-driven encoders,
  and a hand-written MLP — so it needs no external ML runtime.

---

## 3. Install (conda → git → Isaac Sim → Isaac Lab → this repo)

Isaac Sim and Isaac Lab are installed **via pip inside a conda environment**. Do the steps in order.
Exact package versions depend on your Isaac Sim release — this guide follows the official
[Isaac Lab pip-install docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html);
open that page and match the versions to **Isaac Sim 5.1** if they differ from the examples below.

### 3.1 Install conda (Miniconda) and git

**Windows**
```powershell
# Option A — one-liner with winget (Windows 10/11):
winget install -e --id Anaconda.Miniconda3
winget install -e --id Git.Git
# Option B — manual: download and run the installers:
Miniconda : https://www.anaconda.com/download  (or https://docs.conda.io/en/latest/miniconda.html)
Git       : https://git-scm.com/download/win
After installing, use the "Anaconda Prompt" so `conda` is on PATH.
```

**Ubuntu (20.04 / 22.04)**
```bash
# git + build tools:
sudo apt-get update && sudo apt-get install -y git cmake build-essential
# Miniconda:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init            # then restart the terminal
```

### 3.2 Create and activate a conda environment (Python 3.11)
```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

### 3.3 Install Isaac Sim via pip
```bash
pip install --upgrade pip
# Replace the version with the one matching your Isaac Sim release (example: 5.1.0):
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

```

### 3.4 Install a CUDA-enabled PyTorch
Pick the CUDA build matching your driver (example uses CUDA 12.8). See the official docs for the exact
torch version paired with your Isaac Sim release.
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# Verifying the Isaac Sim installation
isaacsim
```
On first launch Isaac Sim compiles shaders/extensions — that first run can take several minutes.

### 3.5 Install Isaac Lab
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
# Windows:
isaaclab.bat --install rsl_rl

## Optional
isaaclab.bat --install rl_games
isaaclab.bat --install skrl
isaaclab.bat --install sb3

## run this to avoid the crash
pip install --no-deps --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps --force-reinstall "tensordict==0.8.*"
pip install --no-deps typing_extensions==4.12.2 psutil==5.9.8
```

### 3.6 Clone THIS repository and install the extension
```bash
git clone https://github.com/LinhTranUlsan/balancing-robot-pai.git
cd balancing-robot-pai

# Editable install of the TWIP extension into the same conda env:
python -m pip install -e source/Twip_Rsl_v2

# Verify the task registered:
python scripts/list_envs.py        # should list: Template-Twip-Rsl-V2-v0
```

> From here on, run every `scripts/` command **with `env_isaaclab` activated** (that env now contains
> Isaac Sim). If you did NOT install Isaac Sim into the conda env, prefix scripts with the Isaac Lab
> launcher instead: `<path-to-IsaacLab>\isaaclab.bat -p <script>` (Windows) /
> `<path-to-IsaacLab>/isaaclab.sh -p <script>` (Linux).

---

## 4. Train → get an ONNX policy

```bash
# Train (headless is fastest). Checkpoints go to logs/rsl_rl/twip_rsl_v2/<timestamp>/
python scripts/rsl_rl/train.py --task=Template-Twip-Rsl-V2-v0 --headless
# Or show the GUI
python scripts/rsl_rl/train.py --task=Template-Twip-Rsl-V2-v0
# Or change the number of simulation environments
python scripts/rsl_rl/train.py --task=Template-Twip-Rsl-V2-v0 --num_envs=16 # Remove --headless to show the GUI

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

## 5. Deploy the policy on-device (`deploy/`)

`deploy/` is a self-contained **PlatformIO** project (C/C++). `src/main.cpp` runs the policy with a
**hand-written MLP forward** — no TensorFlow/ONNX runtime on the MCU — reading the network weights from
`include/policy_weights.h`. That header defines `POLICY_OBS_DIM` (2: `[pitch, rate]`), `POLICY_ACT_DIM`
(1 or 2 wheel commands), and the layer weights (this build: `2 → 32 → 32 → 1`, ELU).

**To update the on-device policy**, regenerate `deploy/include/policy_weights.h` from a new ONNX with
the bundled converter `deploy/export_policy_header.py`:
1. Export the policy to ONNX (Section 4).
2. Convert ONNX → C header (needs `pip install onnx onnxruntime numpy`):
   ```bash
   python deploy/export_policy_header.py \
       --model <path>/policy.onnx \
       --out   deploy/include/policy_weights.h --verify
   ```
   `--verify` cross-checks the generated C weights against ONNX Runtime; `--check` prints a couple of
   sample outputs to compare with the firmware's boot self-test.
3. Rebuild & upload (Section 6). The firmware enforces `POLICY_OBS_DIM == 2` at **compile time** and
   supports 1 or 2 actions; a different obs size stops the build with a clear `#error`.

---

## 6. Build & flash the ESP32-S3 (PlatformIO)

From the `deploy/` folder (PlatformIO CLI, or the VS Code PlatformIO buttons):
```bash
cd deploy
pio run                 # compile
pio run -t upload       # flash the ESP32-S3 over USB
pio device monitor      # serial monitor @ 115200 (pitch / rate / motor commands / encoders)
```
- Board & flash settings live in `platformio.ini` (board `esp32-s3-devkitm-1`, 4 MB flash,
  USB-CDC-on-boot).
- **At boot the robot calibrates the IMU — hold it upright and still for ~2 s** until the serial log
  prints the offset. Re-run calibration any time with the `c` command.
- Runs a **100 Hz** super-loop (`CONTROL_HZ = 100`, matching training). A built-in serial tuner lets
  you flip motor/gyro signs and set the pitch trim live — type `?` in the monitor to list commands
  (e.g. `m -1`, `j -1`, `k -1`, `s <deg>`, `t` motor test, `x` / `o` stop / go).

---

## 7. Troubleshooting (common errors)

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: No module named 'Twip_Rsl_v2'` | You skipped the editable install. Run `python -m pip install -e source/Twip_Rsl_v2` **in the Isaac Lab env**. |
| `FileNotFoundError: ...TwoWheel.urdf` | Keep the repo layout intact, or set env var `TWIP_URDF=/abs/path/TwoWheel.urdf`. |
| `list_envs.py` shows nothing / task missing | Import side-effect didn't run — reinstall the extension; make sure you're using the Isaac Lab interpreter. |
| rsl-rl version error on `train.py` | Install the exact version the script prints (`rsl-rl-lib>=3.0.1`). |
| Isaac Sim crashes / GPU errors | Wrong Isaac Sim version or driver — use 5.1 (or 4.5/5.0) with a supported RTX driver. |
| PlatformIO: build fails / toolchain not found | Run commands from inside `deploy/` so it reads `platformio.ini`; let the first build install the ESP32 platform. |
| Build error: `#error "Firmware lap obs 2 chieu ..."` | `policy_weights.h` has `POLICY_OBS_DIM != 2` — re-export a 2-obs `[pitch, rate]` policy, or adapt `assembleObs()` in `src/main.cpp`. |
| Robot balances in sim but not on hardware | Hold it still during the boot IMU calibration, use the serial tuner (`?`) to fix motor/gyro sign and pitch trim, and verify wiring against the pin table above. |

---

## 8. License & credits

- Built on the [Isaac Lab](https://github.com/isaac-sim/IsaacLab) external-extension template
  (BSD-3-Clause headers retained on upstream-derived files).
- TWIP RL env, firmware, and ONNX→C tooling by the project author (PAI, University of Ulsan).

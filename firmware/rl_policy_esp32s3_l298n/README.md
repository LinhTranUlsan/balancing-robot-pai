# RL policy ON-BOARD — L298N + 2 động cơ (nhúng thẳng vào ESP32-S3, KHÔNG cần PC)

Phần cứng: cầu H **L298N** + **2 động cơ** (2 bánh chung trục) + BMI160 + 2 encoder.
**Chế độ 3 (on-board)** — mạng MLP dịch thành trọng số C trong `policy_weights.h`, chạy ngay
trên chip, không TFLite / không ONNX runtime.

```
.onnx (PC)                              ESP32-S3 (tự trị, 200 Hz)
   │  export_policy_header.py             ┌───────────────────────────────────┐
   └────────▶ policy_weights.h ──────▶    │ IMU → lọc góc → MLP → u → L298N ×2 │
              (nạp cùng firmware)         └───────────────────────────────────┘
```

Observation vẫn **2 chiều** `[pitch (rad), rate (rad/s)]` (IMU-only, chưa dùng encoder trong
policy), Action vẫn **1 giá trị** `u ∈ [-1,1]` → áp **cùng lúc cho cả 2 motor**.

---

## ⚠️ Đã tinh chỉnh dựa trên bộ PID chạy tốt của bạn (đọc log → vì sao xe ngã)

Log bạn gửi: `pitch` tăng dần 6°→9°→13°→**27°** trong khi `u` đã bão hoà `-0.99`, rồi mới giật
lại và dao động → xe "tiến thẳng rồi đâm đầu". Nguyên nhân + cách sửa (rút từ `L298N_package`):

| Vấn đề (bản cũ) | Sửa (bản này) |
|---|---|
| **Policy quá yếu**: bắt chước PD `Kp≈5/rad` — mềm hơn PID của bạn ~5 lần → để góc lớn rồi mới phản ứng | `policy_weights.h` sinh từ **`example_policy_strong.onnx`** (`kp=24, kd=1.0`). Độ dốc thực gần gốc (do tanh) ≈ **Kp 0.58/độ, Kd 0.025/độ·s**, bão hoà ~1.7° → hơi **cứng + damp nhiều hơn** PID cascade (0.49/0.014) một chút = thiên về ổn định, không dao động. Muốn dịu: `--kp 20 --kd 0.6` |
| **Vòng lặp 60 Hz** — chậm cho một controller cứng | **200 Hz** (`CONTROL_HZ`) — khớp vòng góc 200 Hz của PID; policy PD tĩnh nên chạy nhanh là an toàn |
| **`driveU` cũ nhảy thẳng lên `MIN_DUTY`** khi `\|u\|>deadband` → bước torque → xe rung/đi lung tung quanh cân bằng | **MotorCommandShaper** (port từ package): dưới breakaway dùng **pulse-density** (torque trung bình ~ lệnh) + **trần duty gần cân bằng (0.42)** + **coast khi đảo chiều** |
| **Duty yếu** `MIN=0.10 / MAX=0.70` | `MIN=0.14 / MAX=0.90` (như package), deadband `0.04` (nuốt bias tĩnh ~0.03 của MLP) |

Các thay đổi trên khiến bản RL này cư xử gần giống bộ PID cascade đang chạy ổn của bạn — mục tiêu
là **chứng minh toàn bộ pipeline nhúng ONNX giữ được cân bằng**, để sau này thay policy train thật.

---

## Đấu nối — KHỚP SƠ ĐỒ CHÂN THỰC TẾ của robot

| Chân ESP32-S3 | Nối tới | Ý nghĩa |
|---|---|---|
| **GPIO6 / GPIO5** | BMI160 **SDA / SCL** | I2C IMU (⚠️ xem cảnh báo bên dưới) |
| **GPIO7 / GPIO8** | L298N **IN1 / IN2** | chiều **motor A (bánh trái)** |
| **GPIO11** | L298N **ENA** | PWM tốc độ motor A |
| **GPIO9 / GPIO10** | L298N **IN3 / IN4** | chiều **motor B (bánh phải)** |
| **GPIO12** | L298N **ENB** | PWM tốc độ motor B |
| **GPIO1 / GPIO2** | Encoder **trái A / B** | quadrature (đọc telemetry, chưa vào obs) |
| **GPIO4 / GPIO13** | Encoder **phải A / B** | quadrature (đọc telemetry, chưa vào obs) |
| GPIO21 | LED RGB onboard | trạng thái |

> ⚠️ **SDA/SCL ĐẢO so với các firmware cũ.** Sơ đồ của bạn là **SDA=GPIO6, SCL=GPIO5**; các firmware
> DRV8871 cũ dùng SDA=5/SCL=6. Bản này theo **sơ đồ mới (6/5)** — trùng đúng `L298N_package`
> (`i2cSda=6, i2cScl=5`). Nếu self-test báo không thấy BMI160 thì đảo lại `SDA_PIN`/`SCL_PIN`.

> ⚠️ **Encoder đã đấu dây nhưng CHƯA vào observation.** Policy vẫn 2 chiều (IMU-only) đúng như bạn
> dặn. Firmware **đọc encoder chạy nền** (đếm xung + vận tốc bánh) in ra telemetry `encL/encR/velL/velR`
> để kiểm tra encoder sống, và sẵn cho obs 4 chiều sau này (xem `assembleObs()`). `ENC_TICKS_PER_REV`
> đang để **332** (≈ CPR x4 = 1327 của package chia 4, vì firmware này đếm x1); sửa cho đúng nếu cần.

**Nguồn L298N:** nguồn động cơ vào `VS/+12V`, **GND chung** với ESP32; logic `+5V`. **Tháo jumper
ENA/ENB** để PWM điều khiển tốc độ. Logic 3.3V đủ kéo IN/EN của L298N — **không cần level shifter**.
L298N **sụt ~2V**: điện áp motor ≈ nguồn − 2V; motor yếu thì nâng nguồn (9–12V).

---

## Quy trình chạy (từng bước)

### Bước 1 — `policy_weights.h` (đã có sẵn, sinh từ policy khoẻ)
Header **đã có sẵn** trong thư mục này, sinh từ `pc_policy/example_policy_strong.onnx`
(`kp=24, kd=1.0`), **đã VERIFY khớp onnxruntime (sai lệch ~1.8e-7)**. Chỉ chạy lại khi đổi policy:
```bash
python pc_policy/export_policy_header.py \
       --model pc_policy/example_policy_strong.onnx \
       --out   firmware/rl_policy_esp32s3_l298n/policy_weights.h
```
Muốn đổi độ "cứng" của policy ví dụ → sinh lại rồi export:
```bash
python pc_policy/make_example_onnx.py --out pc_policy/example_policy_strong.onnx --kp 24 --kd 1.0
```
`--kp` cao hơn = cứng hơn (phản ứng mạnh ở góc nhỏ). Model phải là **MLP thuần** (Gemm/MatMul +
ELU/ReLU/Tanh/Sigmoid); LSTM/CNN bị từ chối. (Chỉ có `.pt` rsl_rl → dùng `scripts/rsl_rl/play.py`
của Isaac Lab để xuất `policy.onnx` trước.)

### Bước 2 — Nạp firmware
Mở `rl_policy_esp32s3_l298n.ino` trong Arduino IDE (header cùng thư mục dịch kèm tự động).
Board **ESP32S3 Dev Module**, **USB CDC On Boot: Enabled**, core ≥ 3.0. Serial **115200**.

### Bước 3 — Calib + self-test (bánh KHÔNG chạm đất)
Cấp nguồn, **giữ robot thẳng đứng và yên ~2 s** (LED xanh dương) để calib. Sau calib in:
```
# SELF-TEST obs=[0.10, 0.00] -> action=-0.9972 (infer XXus)
```
Giá trị phải ≈ **-0.997** (policy khoẻ). `infer` vài chục µs → dư cho vòng 200 Hz (5 ms).

### Bước 4 — KIỂM TRA CHIỀU ⚠️ (bánh vẫn KHÔNG chạm đất) — **2 việc**
1. **Chiều chống ngã (`m`):** nghiêng robot về trước → **cả 2 bánh quay theo hướng "chạy tới đỡ"**.
   Mặc định `MOTOR_SIGN = -1` (khớp dấu PID của bạn). Nếu đẩy **sai** → gõ `m 1`.
2. **2 bánh cùng chiều (`j`/`k`):** 2 motor lắp đối xứng → cùng lệnh có thể quay ngược nhau (xe xoay
   tại chỗ). Gõ `t` để test; nếu 2 bánh lăn ngược nhau → gõ `k -1` đến khi **cùng hướng**.

Chốt xong chép vào `MOTOR_SIGN`/`MOTOR_A_SIGN`/`MOTOR_B_SIGN` trong `.ino`.

### Bước 5 — Đặt xuống thử (luôn đỡ tay)
- Nếu xe **trôi đều một chiều** dù đang đứng: chỉnh **trim** `s <độ>` (vd `s -1.5`) cho tới khi hết
  trôi — giống `targetDeg` của PID (bù CoM không nằm đúng trên trục). Mặc định `0`.
- Rung/giật mạnh lúc đầu: giảm ga `a 0.7`, hạ trần `l 0.8`, hoặc tăng deadband `b 0.04`.
- `x` = dừng khẩn cấp, `o` = bật lại. Nghiêng > 45° tự COAST cả 2 motor.

---

## Lệnh Serial (115200)
```
m 1 / m -1   chiều chống ngã (MOTOR_SIGN)   g 1 / g -1   đảo dấu gyro
j 1 / j -1   đảo riêng bánh TRÁI  (A)        k 1 / k -1   đảo riêng bánh PHẢI (B)
s -1.5       trim góc cân bằng (độ)          a 0.7        scale action (0..1)
n / l / b    MIN_DUTY / MAX_DUTY / deadband  c            calib lại (giữ thẳng)
t            test 2 motor 2 chiều            x / o        DỪNG KHẨN / bật lại
?            in tham số + thời gian inference
```
Chỉnh qua Serial chỉ **tạm thời** (mất khi reset) — chốt xong chép vào `USER CONFIG` trong `.ino`.

---

## Tốc độ lấy mẫu — vì sao 200 Hz (đã kiểm tra)

| Thứ | Giá trị | Lý do |
|---|---|---|
| **Vòng điều khiển** | **200 Hz** (`CONTROL_HZ`, `LOOP_US=5000`) | Khớp **vòng góc 200 Hz** của PID cascade chạy tốt. Policy ví dụ là **PD tĩnh (memoryless)** nên chạy nhanh hơn 60 Hz là **an toàn** và ổn định hơn (controller cứng cần vòng nhanh). |
| **IMU (BMI160 ODR)** | 400 Hz | Nhanh gấp đôi vòng lặp → luôn có mẫu mới, không dùng lại mẫu cũ. |
| **MLP inference** | ~0.1 ms | ≪ chu kỳ 5 ms → dư thời gian. |
| **PWM L298N** | 20 kHz (chân ENABLE) | Trên ngưỡng nghe. Motor yếu/L298N nóng → hạ `PWM_FREQ` ~1000–8000. |

> ⚠️ **Khi thay POLICY TRAIN THẬT từ Isaac Lab: phải đặt `CONTROL_HZ = 60`.** Env sim chạy
> `dt=1/120 × decimation=2 = 1/60 s`, policy train thật (nhất là loại có trạng thái/thời gian) **bắt
> buộc** chạy đúng tần số lúc train. 200 Hz chỉ đúng cho policy PD tĩnh ví dụ này (không phụ thuộc thời gian).

---

## Ràng buộc & lưu ý

- **Obs = 2 chiều `[pitch (rad), rate (rad/s)]`** đúng như lúc train. Model khác số chiều →
  firmware **báo lỗi khi biên dịch** (`#error`), sửa `assembleObs()` (vd thêm encoder cho 4 chiều).
- **Đây là policy PD ví dụ**, đã chỉnh khớp gain PID cascade của bạn → kỳ vọng đứng gần bằng PID.
  Nó **không phải** policy RL train thật; chất lượng đầy đủ cần train trên Isaac Lab.
- **Reality gap khi train thật:** sim map action → lực tuyến tính; thực có `MIN_DUTY` (bù ma sát) +
  trần + sụt áp L298N + deadzone → nên bật **domain randomization** (deadzone/độ trễ/nhiễu motor).
- **Không drift chuẩn nếu chưa dùng encoder:** giống PID IMU-only, xe có thể trôi chậm. Trim `s` giảm
  trôi; muốn khử hẳn thì train obs 4 chiều (thêm cart_pos/cart_vel từ encoder) — hạ tầng đã sẵn.
- Muốn tune/thử policy nhanh mà không nạp lại chip → dùng chế độ **PC-bridge** (`rl_bridge`), chốt
  policy rồi mới nhúng on-board như bản này.

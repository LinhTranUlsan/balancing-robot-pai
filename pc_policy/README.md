# PC-in-the-loop: chạy policy ONNX trên PC, ESP32 làm cầu cảm biến/động cơ

Kiến trúc: **PC chạy mạng (ONNX), ESP32 đọc IMU + lái motor và truyền cảm biến về PC.**

```
        ┌─────────── ESP32-S3 (rl_bridge, 60Hz) ───────────┐        ┌──────── PC ────────┐
IMU ───▶ đọc ─▶ gửi "O pole_pos pole_vel cart_pos cart_vel" ──USB──▶ assemble_obs (2 chiều)
                                                                     │  ONNX inference
DRV8871 ◀─ map u→PWM ◀───────── nhận "A <u>" ◀──────────────USB───────  action → u∈[-1,1]
         (watchdog: mất lệnh >100ms → COAST)
```

---

## ⚠️ CÓ 3 CHẾ ĐỘ — ĐỪNG NHẦM FIRMWARE

| | **Chế độ 1: PID độc lập** | **Chế độ 2: RL qua PC (mục này)** | **Chế độ 3: RL on-board** |
|---|---|---|---|
| Firmware nạp ESP32 | `balancing_pid_esp32s3.ino` | `rl_bridge_esp32s3.ino` | `rl_policy_esp32s3.ino` |
| Ai tính lệnh? | **ESP32 tự tính PID** | **PC chạy ONNX**, ESP32 chỉ là cầu | **ESP32 tự chạy MLP** (trọng số nhúng) |
| Cần PC chạy? | Không (chạy độc lập) | **Có** — phải chạy `run_policy_serial.py` | Không (PC chỉ để sinh header) |
| Dùng khi | cân bằng ngay, không cần sim | test/tune policy đã train (ONNX) | policy đã chốt, chạy tự trị |

> File `balancing_pid_esp32s3.ino` **KHÔNG liên quan** tới ONNX. Muốn chạy ONNX qua PC thì nạp **`rl_bridge_esp32s3.ino`**; muốn nhúng ONNX vào chip (không cần PC) xem [../firmware/rl_policy_esp32s3/README.md](../firmware/rl_policy_esp32s3/README.md) — dùng `export_policy_header.py` (trong thư mục này) để sinh `policy_weights.h`.

---

## Robot hiểu lệnh `u ∈ [-1,1]` thế nào? (CÓ, đúng)

Cả 2 firmware map `u` ra PWM **giống hệt nhau**:
```
u = 0                      -> |u| < U_DEADBAND  -> COAST (motor tắt)
0 < |u| ≤ 1                -> duty = MIN_DUTY + |u|·(MAX_DUTY − MIN_DUTY)
dấu của u                  -> chiều quay (IN1 hay IN2)
```
- `|u|=1` → duty = **MAX_DUTY**;  `|u|` nhỏ (trên deadband) → duty = **MIN_DUTY**.
- Nên **toàn dải [-1,1] được dùng đúng**: dấu = chiều, độ lớn = mức ga (giữa MIN và MAX).
- MIN/MAX duty **không mâu thuẫn** với [-1,1] — chúng chỉ định *cách diễn giải* độ lớn ra PWM thực (bù ma sát ở dưới, chặn trần ở trên).

Giá trị hiện tại: **rl_bridge** dùng `MIN=0.10, MAX=1.00` (đủ lực cho RL); **PID** dùng `MAX=0.70` (êm hơn). Đổi được trong từng file (bridge sửa hằng số; PID sửa live qua Serial `n`/`l`).

> Action từ `example_policy.onnx` đã qua `tanh` nên luôn nằm trong [-1,1] → khớp sẵn. `run_policy_serial.py` chỉ clip an toàn rồi gửi thẳng làm `u`.

---

## Cách chạy CHẾ ĐỘ RL (từng bước)

### Bước 0 — Cài đặt (PC)
```bash
pip install onnxruntime onnx numpy pyserial     # torch chỉ cần nếu tự sinh lại ONNX
```
Observation hệ thực hiện tại (chưa encoder) = **2 chiều IMU: `[pole_pos (rad), pole_vel (rad/s)]`**.

### Bước 1 — Kiểm tra ONNX ví dụ (đã tạo sẵn)
```bash
python inspect_onnx.py --model example_policy.onnx
```
Kết quả mong đợi (xem `example_dump.txt`): input `obs [batch,2]`, output `actions [batch,1]`, và `obs=[0.1,0] → action≈-0.59` (đúng chiều). *(Tạo lại nếu cần: `python make_example_onnx.py`.)*

### Bước 2 — Nạp firmware cầu cho ESP32
Nạp `../firmware/rl_bridge_esp32s3/rl_bridge_esp32s3.ino` (Arduino IDE, ESP32S3 Dev Module, USB CDC On Boot = Enabled). Lúc khởi động **giữ robot thẳng đứng ~2s** để calib. ESP32 bắt đầu phát `O ...` ở 60 Hz.

### Bước 3 — Chạy vòng lặp (KHÔNG cần sửa code)
**Bánh KHÔNG chạm đất lần đầu.** Thay `COM6` bằng cổng của bạn:
```bash
python run_policy_serial.py --port COM6 --model example_policy.onnx
```
- Script **tự khớp** `example_policy.onnx` (obs 2 chiều) — không phải sửa gì. Nếu số chiều lệch, nó **báo lỗi ngay** thay vì chạy sai.
- Xem log: tần số (Hz) + độ trễ (ms) + `u`. Nghiêng robot bằng tay → motor phải phản ứng **chống ngã**.

### Bước 4 — Kiểm chiều & đặt xuống thử
- Nếu motor đẩy **sai chiều** → sửa `action_to_motor_u()` thành `return -u`, **hoặc** đảo `MOTOR_SIGN` trong firmware bridge. (Chỉ chỗ này mới cần đụng.)
- Đúng chiều rồi mới đặt xuống, luôn có tay đỡ. Ctrl+C (hoặc mất kết nối >100ms) → motor tự dừng.

---

## Khi nào MỚI phải sửa `run_policy_serial.py`?
Chỉ khi bạn **đổi env/observation** (sau này train thật trên Isaac Lab):
- **Thêm encoder + train cartpole 4 chiều** → mở `assemble_obs()` thêm `cart_pos, cart_vel` (đã có dòng mẫu ghi sẵn), và cho ESP32 điền 2 giá trị đó từ encoder.
- **Motor sai chiều** → `action_to_motor_u` đổi dấu.
Ngoài 2 việc đó, với `example_policy.onnx` hiện tại thì **chạy thẳng, không sửa gì**.

## Giao thức serial (ASCII, dễ debug)
| Chiều | Định dạng | Ý nghĩa |
|---|---|---|
| ESP32 → PC | `O <pole_pos> <pole_vel> <cart_pos> <cart_vel>\n` | obs thô (radian, rad/s, m, m/s) |
| PC → ESP32 | `A <u>\n` | lệnh motor u∈[-1,1] |
| ESP32 → PC | `# ...` | log/ghi chú (PC bỏ qua) |

## Lưu ý quan trọng
1. **Độ trễ + dây USB**: ESP32-S3 dùng USB-CDC gốc (~1–3 ms), 60 Hz khả thi. Xem số `ms` trong log; > ~8 ms là rủi ro cho cân bằng. Dây USB cũng ràng buộc robot.
2. **Obs phải khớp TUYỆT ĐỐI** với lúc train (thứ tự, đơn vị **radian**, scale). Script tự kiểm tra *số chiều*; nhưng thứ tự/đơn vị là trách nhiệm của bạn.
3. **`cart_pos`/`cart_vel` = 0** (chưa encoder). Obs hiện dùng 2 chiều IMU nên không ảnh hưởng. Khi train 4 chiều mới cần encoder.
4. **Normalizer** (nếu env thật bật) đã nhúng trong ONNX → không chuẩn hóa lại ở PC.
5. **Reality gap** cho policy thật: sim map action→torque tuyến tính, còn thực có MIN_DUTY (bù ma sát) + trần → nên **mô phỏng deadzone/độ trễ/nhiễu motor trong sim (domain randomization)** khi train.

## File trong gói
- `make_example_onnx.py` — sinh `example_policy.onnx` (MLP 2→32→32→1 bắt chước PD).
- `example_policy.onnx` — **ONNX ví dụ đã tạo sẵn** (obs 2, action 1), chạy được ngay.
- `inspect_onnx.py` — dump cấu trúc + chạy thử (kết quả mẫu ở `example_dump.txt`).
- `run_policy_serial.py` — vòng lặp PC (obs→ONNX→u). Chạy thẳng, tự kiểm tra số chiều.
- `example_dump.txt` — kết quả dump đã xác thực.
- `../firmware/rl_bridge_esp32s3/` — firmware cầu cho ESP32 (chế độ RL).
- `../firmware/balancing_pid_esp32s3/` — firmware PID độc lập (chế độ 1, KHÔNG dùng ONNX).

Điều quan trọng nhất cần hiểu đúng: ONNX không nạp xuống ESP32
Kiến trúc hiện tại là PC-in-the-loop: file .onnx chạy trên PC bằng onnxruntime, còn ESP32 chỉ nạp firmware "cầu" (rl_bridge_esp32s3.ino) một lần duy nhất. Sau đó mỗi lần đổi policy, bạn chỉ thay file .onnx trên PC, không đụng gì tới ESP32 nữa. Vòng lặp 60 Hz:

ESP32 đọc BMI160 → gửi O <pole_pos> <pole_vel> 0 0 (radian) qua USB
PC (run_policy_serial.py) lắp obs 2 chiều → chạy ONNX → gửi A <u> với u ∈ [-1,1]
ESP32 map u → PWM cho DRV8871; mất lệnh >100 ms hoặc nghiêng >45° thì tự tắt motor
Quy trình chạy đúng (đã đối chiếu code, khớp với README)
PC: pip install onnxruntime onnx numpy pyserial
Kiểm tra ONNX trước: python inspect_onnx.py --model example_policy.onnx — phải thấy input [batch, 2], output [batch, 1], và obs=[0.1, 0] → action ≈ -0.59 (khớp example_dump.txt)
Nạp firmware cầu: rl_bridge_esp32s3.ino (ESP32S3 Dev Module, USB CDC On Boot = Enabled, core ≥ 3.0). Lúc khởi động giữ robot thẳng đứng và yên ~2 giây để calib — nếu bỏ qua bước này, góc gửi lên PC bị lệch offset và policy sẽ điều khiển sai
Bánh không chạm đất, chạy: python run_policy_serial.py --port COMx --model example_policy.onnx — xem log Hz/latency, nghiêng robot bằng tay và xác nhận motor quay chống lại chiều ngã
Đúng chiều rồi mới đặt xuống đất, luôn có tay đỡ. Ctrl+C hoặc rút dây → watchdog tự COAST
Pipeline này nhất quán: đơn vị radian khớp hai đầu, script tự kiểm tra số chiều obs và báo lỗi sớm nếu lệch, an toàn (watchdog + giới hạn nghiêng) đều có. Nhưng có 3 điểm bạn cần biết trước khi chạy:

⚠️ Phát hiện 1 — Gần như chắc chắn phải đảo chiều motor ở bước 4
Đây là điểm đáng chú ý nhất. Firmware PID đang chạy được của bạn tính u = +Kp*error + Kd*rate (balancing_pid_esp32s3.ino:267) — tức nghiêng dương → u dương là chiều chống ngã đúng trên phần cứng của bạn (với MOTOR_SIGN=+1). Trong khi đó example_policy.onnx được huấn luyện bắt chước u = -Kp*θ - Kd*ω — nghiêng dương → u âm. Hai firmware dùng cùng cách đo góc, cùng chân, cùng cách map u→PWM, nên nếu robot của bạn cân bằng được với PID hiện tại thì example ONNX sẽ đẩy ngược chiều với cấu hình mặc định. Bước 4 trong README có nói "nếu sai chiều thì đảo", nhưng thực tế nên chuẩn bị tinh thần là sẽ phải đảo: sửa action_to_motor_u() thành return -u trên PC, hoặc đặt MOTOR_SIGN = -1 trong firmware bridge (chọn một trong hai, đừng làm cả hai).

⚠️ Phát hiện 2 — Bảng đấu nối trong README PID sai chân
README.md của firmware PID ghi DRV8871 IN1→GPIO14, IN2→GPIO15, nhưng cả hai file .ino (PID lẫn bridge) đều dùng GPIO7/GPIO8. Nếu bạn đấu dây theo bảng README thì motor sẽ không chạy với cả hai firmware. Nên sửa lại bảng đó thành GPIO7/GPIO8 cho khớp code (package PlatformIO cũ cũng ghi 7/8).

⚠️ Phát hiện 3 — Example ONNX yếu hơn đáng kể so với PID đã tune
Quy đổi về cùng đơn vị: PID thật của bạn Kp = 0.412/độ ≈ 23.6/rad, còn example ONNX bắt chước PD với Kp = 5.0/rad — mềm hơn khoảng 5 lần ở góc nhỏ. Nghĩa là file example này chủ yếu để chứng minh pipeline PC↔ESP32 hoạt động (motor phản ứng đúng chiều, đúng tần số, đúng độ trễ), chứ không đảm bảo robot đứng vững được như PID. Nếu muốn example "khỏe" hơn để thử, có thể sinh lại với gain cao hơn: python make_example_onnx.py --kp 20 --kd 1.0. Ngoài ra bridge dùng MAX_DUTY=1.00 trong khi PID bạn chạy êm ở 0.70 — nếu lần đầu thấy giật mạnh, hạ hằng số MAX_DUTY trong bridge xuống 0.7 cho giống điều kiện đã biết.

Khi có file ONNX thật train từ Isaac Lab
Pipeline giữ nguyên, chỉ thay file .onnx, với điều kiện observation lúc train khớp tuyệt đối với thứ ESP32 gửi lên: 2 chiều [pole_pos, pole_vel] theo radian. Cartpole mặc định trong repo này có obs 4 chiều (thêm cart_pos, cart_vel) — muốn dùng trực tiếp thì phải thêm encoder vào robot và sửa firmware điền 2 giá trị đó, hoặc sửa env trong sim về 2 chiều IMU-only rồi train lại. inspect_onnx.py chạy trước sẽ cho biết ngay model cần bao nhiêu chiều, và run_policy_serial.py sẽ từ chối chạy nếu lệch. Lưu ý thêm từ README (đúng): normalizer nếu có đã nhúng trong ONNX nên PC không chuẩn hóa lại, và khi train thật nên thêm domain randomization mô phỏng deadzone/độ trễ motor vì sim map action→lực tuyến tính còn thực tế có MIN_DUTY bù ma sát.

Tóm lại: tài liệu và code trong 3 folder này nhất quán và quy trình trong pc_policy/README.md là đáng tin để làm theo, chỉ cần lưu ý 3 điểm trên (đặc biệt là chuyện đảo chiều — làm kỹ bước thử trên tay trước khi đặt xuống đất). Nếu bạn muốn, tôi có thể sửa luôn bảng chân cắm sai trong README PID.

f:\Den\2025_HumanoidRobot\0.Simulation\IsaacSim_5.1\Balancing\firmware\rl_bridge_esp32s3\rl_bridge_esp32s3.ino
khoan cho tôi hỏi xíu là tôi dùng cảm biến gia tốc 6 góc thì phải gửi 6 biến tọa độ chứ đúng không, hiện tai tôi thấy gửi có 2 à, nó tính chung lại hả
Đúng như bạn đoán ở câu cuối — nó tính gộp lại. BMI160 là IMU 6 trục (3 trục gia tốc + 3 trục gyro), firmware có đọc đủ cả 6 giá trị thô, nhưng sau đó hợp nhất (sensor fusion) thành đúng 2 con số mà policy cần: góc nghiêng và tốc độ nghiêng.

Chuỗi xử lý trong rl_bridge_esp32s3.ino:

readSensors() (dòng 75-83) đọc 12 byte = 6 giá trị thô, nhưng thực tế chỉ dùng 3: ax, az (gia tốc) và gy (gyro trục Y).
updateAngle() (dòng 86-94) trộn chúng bằng complementary filter:
Gia tốc kế cho biết hướng trọng lực → suy ra góc nghiêng tuyệt đối: atan2(az, ax). Chính xác lâu dài nhưng nhiễu khi robot rung/di chuyển.
Gyro cho tốc độ quay → tích phân ra góc. Mượt tức thời nhưng bị trôi (drift) theo thời gian.
Công thức pitch = 0.98·(pitch + rate·dt) + 0.02·góc_accel lấy ưu điểm của cả hai: gyro lo phần nhanh, accel kéo về đúng chỗ phần chậm.
Kết quả chỉ còn 2 đại lượng vật lý: pole_pos (góc nghiêng, rad) và pole_vel (tốc độ nghiêng, rad/s) → gửi lên PC.
Vì sao chỉ cần 2 chứ không phải 6: robot 2 bánh đồng trục chỉ có một chuyển động duy nhất cần điều khiển — ngã tới/ngã lui (quay quanh trục Y, trong mặt phẳng X-Z). Các trục còn lại (roll, yaw, gia tốc ngang) không giúp gì cho việc cân bằng nên bỏ qua. Quan trọng hơn: input của policy phải khớp observation lúc train trong sim, mà env cartpole định nghĩa obs là [góc, tốc độ góc] chứ không phải dữ liệu cảm biến thô — nên dù cảm biến có 6 hay 9 trục, thứ gửi cho mạng vẫn phải là 2 giá trị đã xử lý này.

(Có hướng khác là train policy ăn thẳng 6 giá trị IMU thô, nhưng khi đó phải mô phỏng cả nhiễu/bias cảm biến trong sim — phức tạp hơn nhiều mà không cần thiết cho bài toán này. Firmware PID của bạn cũng làm y hệt: cân bằng chỉ dựa trên pitch và rate sau filter.)

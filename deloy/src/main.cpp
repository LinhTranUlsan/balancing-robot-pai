#include <Arduino.h>
#include <Wire.h>
#include <BMI160Gen.h>
#include <math.h>
#include <ESP32Encoder.h>

// --- TFLite Micro (thu vien Chirale_TensorFlowLite) ---
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // <- mang g_policy_model (doi ten neu file khac)

// ========================== PHAN CUNG ================================
#define I2C_SDA    6
#define I2C_SCL    5

#define MOTOR_IN1  7
#define MOTOR_IN2  8
#define MOTOR_ENA  11

#define MOTOR2_IN1 9
#define MOTOR2_IN2 10
#define MOTOR2_ENB 12

#define ENCL_A 1
#define ENCL_B 2
#define ENCR_A 3
#define ENCR_B 13

const int PWM_CH1 = 0;
const int PWM_CH2 = 1;
const int      PWM_FREQ = 18000;
const int      PWM_RES  = 8;
const uint32_t MAX_DUTY = (1UL << PWM_RES) - 1UL;
const int bmi160_i2c_addr = 0x69;

const float GYRO_SENS = 131.072f;

// ===================== CAU HINH POLICY ===============================
// >>> 4 THAM SO PHAI KHOP VOI SIM <<<
const uint32_t PID_PERIOD_US = 1000;   // nhip cam bien 1kHz
const int      POLICY_DIV    = 10;     // 1000/10 = 100Hz goi model (dt sim 0.01)
const bool     OBS_IN_RADIAN = true;   // sim dung radian? (gan nhu chac chan)
const float    ACTION_TO_PCT = 100.0f; // action [-1,1] -> % motor
const float    ANGLE_SIGN    = +1.0f;  // dao neu chieu goc nguoc sim
const float    ACTION_SIGN   = -1.0f;  // dao neu xe day cung chieu nga

// ===================== ENCODER / VAN TOC BANH XE =====================
ESP32Encoder encoderL;
ESP32Encoder encoderR;

// So xung tren 1 VONG banh xe SAU quadrature x4 (dung attachFullQuad).
//   = 4 (quadrature) * 11 (PPR) * 30 (ti so hop so) = 1320  (giong sketch test)
// >>> SUA cho dung phan cung cua ban <<<
const float ENC_CPR = 4.0f * 11.0f * 30.0f;

// Dau chieu dem: dao (-1) neu banh tien ma count giam
const float ENC_SIGN_L = -1.0f;
const float ENC_SIGN_R = -1.0f;

const float WHEEL_LPF = 0.3f;          // loc van toc banh (0..1); =1 -> khong loc (giong sim)

// Chuan hoa neu sim co normalize wheel_vel (theo mo ta la KHONG -> = 1)
const float WHEELVEL_SCALE = 1.0f;

// wheel_vel[0] = banh 1 (motor 1 / encoderL), wheel_vel[1] = banh 2 (motor 2 / encoderR)
volatile float wheelVelL = 0.0f;       // rad/s, da loc -> obs wheel_vel[0]
volatile float wheelVelR = 0.0f;       // rad/s, da loc -> obs wheel_vel[1]

// ===================== LOC / AN TOAN =================================
float alpha = 0.99;
const float D_LPF = 0.2;
const float FALL_LIMIT_DEG = 45.0f;

volatile float currentAngle    = 0.0;
volatile float currentGyroRate = 0.0f;
volatile float lastAction1     = 0.0f;
volatile float lastAction2     = 0.0f;
volatile float PCT = 0.0f;
float gyroYOffset = 0;                 // do/s — DUOC TRU khi doc
uint32_t lastMicros = 0;

hw_timer_t*  pidHwTimer    = NULL;
TaskHandle_t pidTaskHandle = NULL;

// ===================== TFLITE MICRO ==================================
constexpr int kArenaSize = 150 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* tin  = nullptr;
static TfLiteTensor* tout = nullptr;

// So I/O MONG DOI (theo spec training). Model phai khop dung day.
const int EXPECT_INPUTS  = 4;   // pitch_angle, pitch_rate, wheel_vel[0], wheel_vel[1]
const int EXPECT_OUTPUTS = 2;   // 2 action / 2 motor

// So I/O THAT doc tu file .tflite luc init
static int nInputs  = 0;
static int nOutputs = 0;

static tflite::MicroMutableOpResolver<12> resolver;

bool modelInit() {
  model = tflite::GetModel(g_policy_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[LOI] Schema version khong khop!");
    return false;
  }

  resolver.AddFullyConnected();
  resolver.AddTanh();
  resolver.AddRelu();
  resolver.AddElu();
  resolver.AddLogistic();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddReshape();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter itp(model, resolver, tensor_arena, kArenaSize);
  interpreter = &itp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[LOI] AllocateTensors that bai -> tang kArenaSize");
    return false;
  }

  tin  = interpreter->input(0);
  tout = interpreter->output(0);

  if (tin->type != kTfLiteFloat32) {
    Serial.println("[LOI] Model khong phai float32!");
    return false;
  }

  // ---- LOP 1: kiem tra SHAPE THAT cua model (float32 -> /4 byte) ----
  nInputs  = tin->bytes  / sizeof(float);
  nOutputs = tout->bytes / sizeof(float);

  Serial.printf("[MODEL] input=%d, output=%d | dims_in:", nInputs, nOutputs);
  for (int i = 0; i < tin->dims->size; i++)  Serial.printf(" %d", tin->dims->data[i]);
  Serial.printf(" | dims_out:");
  for (int i = 0; i < tout->dims->size; i++) Serial.printf(" %d", tout->dims->data[i]);
  Serial.println();

  if (nInputs != EXPECT_INPUTS || nOutputs != EXPECT_OUTPUTS) {
    Serial.printf("[LOI] Model I/O SAI! Can %d vao / %d ra, nhung model la %d vao / %d ra.\n",
                  EXPECT_INPUTS, EXPECT_OUTPUTS, nInputs, nOutputs);
    Serial.println("  -> File .tflite khong khop spec -> export/convert lai model.");
    return false;                 // -> setup() dung o while(1), xe KHONG chay
  }

  Serial.printf("[OK] Model nap: arena dung %u/%d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), kArenaSize);
  return true;
}

// Goi model: obs[nObs] -> out[2]. Tu bao ve theo shape THAT cua model.
//   - Sai so obs (nObs != nInputs)  -> tra 0 (cat motor)
//   - Invoke loi / NaN / inf        -> tra 0 (cat motor)
// ---- LOP 2: chan khong cho tinh tren input rac ----
void policyInfer(const float* obs, int nObs, float out[2]) {
  out[0] = 0.0f;
  out[1] = 0.0f;

  if (nObs != nInputs) return;                 // so obs khong khop -> cat motor

  for (int i = 0; i < nInputs; i++) tin->data.f[i] = obs[i];

  if (interpreter->Invoke() != kTfLiteOk) return;

  int n = (nOutputs < 2) ? nOutputs : 2;
  for (int i = 0; i < n; i++) {
    float y = tout->data.f[i];
    out[i] = (isnan(y) || isinf(y)) ? 0.0f : y; // chan gia tri rac ra moto
  }
}

// ---------------- PWM + L298 (ENB = PWM, IN1/IN2 = chieu) ------------
void pwmSetup() {
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);

  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(MOTOR2_IN1, LOW);
  digitalWrite(MOTOR2_IN2, LOW);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  // core 3.x: gan PWM theo PIN (sua loi symbol MOTOR_ENB/PWM_CH khong ton tai)
  ledcAttach(MOTOR_ENA,  PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR2_ENB, PWM_FREQ, PWM_RES);
#else
  ledcSetup(PWM_CH1, PWM_FREQ, PWM_RES);
  ledcAttachPin(MOTOR_ENA, PWM_CH1);
  ledcSetup(PWM_CH2, PWM_FREQ, PWM_RES);
  ledcAttachPin(MOTOR2_ENB, PWM_CH2);
#endif
}

void pwmWriteDuty(uint32_t duty, const int PWM) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  // core 3.x ghi theo PIN -> map tu kenh sang chan
  int pin = (PWM == PWM_CH1) ? MOTOR_ENA : MOTOR2_ENB;
  ledcWrite(pin, duty);
#else
  ledcWrite(PWM, duty);
#endif
}

void setThrottle(uint8_t in1, uint8_t in2, uint8_t PWM, int pct) {
  pct = constrain(pct, -100, 100);
  if (pct == 0) {                       // dung: cat PWM, motor coast
    pwmWriteDuty(0, PWM);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    return;
  }

  // Bu vung chet L298 + motor
  const uint32_t MIN_DUTY = 150;
  uint32_t duty = MIN_DUTY + (uint32_t)(abs(pct) * (MAX_DUTY - MIN_DUTY) / 100);

  if (pct > 0) {                        // tien
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else {                              // lui
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
  pwmWriteDuty(duty, PWM);
}

// ---------------- ISR: chi danh thuc task ----------------
void IRAM_ATTR onPidTimer() {
  BaseType_t hpw = pdFALSE;
  vTaskNotifyGiveFromISR(pidTaskHandle, &hpw);
  portYIELD_FROM_ISR(hpw);
}

// ---------------- Mot chu ky: IMU 1kHz + model 100Hz ----------------
void runControlOnce() {
  uint32_t now = micros();
  float dt = (now - lastMicros) * 1e-6f;
  lastMicros = now;
  if (dt <= 0 || dt > 0.05f) dt = PID_PERIOD_US * 1e-6f;

  int ax, ay, az, gx, gy, gz;
  BMI160.readMotionSensor(ax, ay, az, gx, gy, gz);

  // thong nhat 131.072 va TRU offset da do luc calib
  float gyroY_actual = gy / GYRO_SENS - gyroYOffset;
  static float gyroFilt = 0.0f;
  gyroFilt = (1.0f - D_LPF) * gyroFilt + D_LPF * gyroY_actual;
  currentGyroRate = gyroFilt;

  float accelAngle = atan2f((float)ax, fabsf((float)az)) * 180.0f / M_PI;
  float angle = alpha * (currentAngle + gyroY_actual * dt)
              + (1.0f - alpha) * accelAngle;
  currentAngle = angle;

  if (fabsf(angle) > FALL_LIMIT_DEG) {
    setThrottle(MOTOR_IN1, MOTOR_IN2, PWM_CH1, 0);
    setThrottle(MOTOR2_IN1, MOTOR2_IN2, PWM_CH2, 0);
    return;
  }

  static int div_cnt = 0;
  if (++div_cnt < POLICY_DIV) return;   // giua 2 lan: giu action cu
  div_cnt = 0;

  // ---------- VAN TOC 2 BANH tu encoder (nhip policy ~100Hz) ----------
  static int64_t lastCntL = 0, lastCntR = 0;
  static uint32_t lastVelUs = 0;
  static float wLf = 0.0f, wRf = 0.0f;   // van toc da loc (rad/s)

  int64_t cntL = (int64_t)(ENC_SIGN_L) * (int64_t)encoderL.getCount();
  int64_t cntR = (int64_t)(ENC_SIGN_R) * (int64_t)encoderR.getCount();

  uint32_t vNow = micros();
  float vdt = (lastVelUs == 0) ? (POLICY_DIV * PID_PERIOD_US * 1e-6f)
                               : (vNow - lastVelUs) * 1e-6f;
  lastVelUs = vNow;                      // <-- cap nhat moc thoi gian (da vá)
  if (vdt <= 0.0f) vdt = POLICY_DIV * PID_PERIOD_US * 1e-6f;

  float dL = (float)(cntL - lastCntL);
  float dR = (float)(cntR - lastCntR);
  lastCntL = cntL;
  lastCntR = cntR;

  // counts -> rad/s cho TUNG banh
  float wL = (dL / ENC_CPR) * 2.0f * (float)M_PI / vdt;
  float wR = (dR / ENC_CPR) * 2.0f * (float)M_PI / vdt;

  wLf = (1.0f - WHEEL_LPF) * wLf + WHEEL_LPF * wL;
  wRf = (1.0f - WHEEL_LPF) * wRf + WHEEL_LPF * wR;
  wheelVelL = wLf;
  wheelVelR = wRf;

  // ---------- OBSERVATION (KHOP SIM: 4 bien) ----------
  //   obs[0] = pitch_angle  (rad)
  //   obs[1] = pitch_rate   (rad/s)
  //   obs[2] = wheel_vel[0] (rad/s, banh 1 / motor 1 / encoderL)
  //   obs[3] = wheel_vel[1] (rad/s, banh 2 / motor 2 / encoderR)
  float pitch_angle, pitch_rate;
  if (OBS_IN_RADIAN) {
    pitch_angle = ANGLE_SIGN * angle    * (float)M_PI / 180.0f;
    pitch_rate  = ANGLE_SIGN * gyroFilt * (float)M_PI / 180.0f;
  } else {
    pitch_angle = ANGLE_SIGN * angle;
    pitch_rate  = ANGLE_SIGN * gyroFilt;
  }

  float obs[4];
  obs[0] = pitch_angle;
  obs[1] = pitch_rate;
  obs[2] = wheelVelL / WHEELVEL_SCALE;   // wheel_vel[0]
  obs[3] = wheelVelR / WHEELVEL_SCALE;   // wheel_vel[1]

  float action[2];
  policyInfer(obs, EXPECT_INPUTS, action);   // dua dung 4 obs
  action[0] = constrain(action[0], -1.0f, 1.0f);
  action[1] = constrain(action[1], -1.0f, 1.0f);
  lastAction1 = action[0];
  lastAction2 = action[1];

  int pct1 = (int)constrain(ACTION_SIGN * action[0] * ACTION_TO_PCT,
                           -100.0f, 100.0f);
  int pct2 = (int)constrain(ACTION_SIGN * action[1] * ACTION_TO_PCT,
                           -100.0f, 100.0f);
  PCT = pct1;
  setThrottle(MOTOR_IN1, MOTOR_IN2, PWM_CH1, pct1);
  setThrottle(MOTOR2_IN1, MOTOR2_IN2, PWM_CH2, pct2);
}

void controlTask(void* arg) {
  lastMicros = micros();
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    runControlOnce();
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(I2C_SDA, I2C_SCL);       // pull-up NGOAI 2.2k-4.7k
  Wire.setClock(400000);
  pwmSetup();
  setThrottle(MOTOR_IN1, MOTOR_IN2, PWM_CH1, 0);
  setThrottle(MOTOR2_IN1, MOTOR2_IN2, PWM_CH2, 0);

  // --- Encoder ---
  ESP32Encoder::useInternalWeakPullResistors = puType::up;  // neu loi: doi thanh = UP;
  encoderL.attachFullQuad(ENCL_A, ENCL_B);   // x4 -> ENC_CPR tinh theo x4
  encoderR.attachFullQuad(ENCR_A, ENCR_B);
  encoderL.clearCount();
  encoderR.clearCount();

  delay(3000);
  Serial.println("\n=== Xe can bang: POLICY (TFLite Micro) ===");

  if (!modelInit()) {
    Serial.println("Khoi tao model that bai — DUNG (xe khong chay).");
    while (1) delay(1000);
  }

  // --- Kiem chung so hoc: doi chieu voi convert_onnx.py cung input ---
  {
    float chk[4] = { -0.5f, 0.5f, 0.0f, 0.0f };  // [angle, rate, wheel0, wheel1]
    float y[2];
    policyInfer(chk, EXPECT_INPUTS, y);
    Serial.printf("[CHECK] input [-0.5, 0.5, 0, 0] -> %+.6f, %+.6f\n", y[0], y[1]);
    Serial.println("  -> Chay convert_onnx.py voi cung input nay de so ket qua.");
  }

  // --- BMI160 ---
  BMI160.begin(BMI160GenClass::I2C_MODE, bmi160_i2c_addr);
  BMI160.setAccelerometerRange(2);
  BMI160.setGyroRange(250);
  BMI160.setAccelRate(BMI160_ACCEL_RATE_800HZ);
  BMI160.setGyroRate(BMI160_GYRO_RATE_800HZ);
  Serial.println("Dang Calib BMI160... Giu co dinh xe.");
  delay(1500);
  BMI160.autoCalibrateAccelerometerOffset(0, 0);
  BMI160.autoCalibrateAccelerometerOffset(1, 0);
  BMI160.autoCalibrateAccelerometerOffset(2,-1);
  BMI160.autoCalibrateGyroOffset();
  delay(200);
  {
    long sum = 0; int ax, ay, az, gx, gy, gz;
    for (int i = 0; i < 500; i++) {
      BMI160.readMotionSensor(ax, ay, az, gx, gy, gz);
      sum += gy; delay(2);
    }
    gyroYOffset = (sum / 500.0f) / GYRO_SENS;   // do/s (se duoc TRU khi doc)
    Serial.printf("GyroY offset = %.4f do/s\n", gyroYOffset);
  }

  // Xoa lai encoder sau calib de bat dau tu 0
  encoderL.clearCount();
  encoderR.clearCount();

  BaseType_t ok = xTaskCreatePinnedToCore(controlTask, "ctrlTask", 8192, NULL,
                        configMAX_PRIORITIES - 1, &pidTaskHandle, 1);
  if (ok != pdPASS) {
    Serial.println("LOI: khong tao duoc controlTask!");
    while (1) delay(1000);
  }

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  pidHwTimer = timerBegin(1000000);
  timerAttachInterrupt(pidHwTimer, &onPidTimer);
  timerAlarm(pidHwTimer, PID_PERIOD_US, true, 0);
#else
  pidHwTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(pidHwTimer, &onPidTimer, true);
  timerAlarmWrite(pidHwTimer, PID_PERIOD_US, true);
  timerAlarmEnable(pidHwTimer);
#endif

  Serial.println("He thong can bang (POLICY/TFLite) da kich hoat!");
}

void loop() {
  Serial.print("Goc:");
  Serial.print(currentAngle);
  Serial.print(",GyroY:");
  Serial.print(currentGyroRate);
  Serial.print(",WvelL:");
  Serial.print(wheelVelL, 2);
  Serial.print(",WvelR:");
  Serial.print(wheelVelR, 2);
  Serial.print(",Action1:");
  Serial.print(lastAction1, 3);
  Serial.print(",Action2:");
  Serial.println(lastAction2, 3);
  Serial.print("PCT:");
  Serial.println(PCT);
  delay(100);
}
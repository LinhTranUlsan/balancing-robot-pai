/* =============================================================================
   RL POLICY ON-BOARD (L298N, 2 MOTORS) — runs the MLP policy DIRECTLY on the ESP32-S3,
   NO PC needed.  For your hardware: L298N H-bridge + 2 motors (2 wheels on one axle).

   Board: ESP32-S3 Zero | IMU: BMI160 | Driver: L298N (dual H-bridge) | 2 DC motor

   >>> TUNED to match your well-working cascade PID (L298N_package). <<<
   Observation = [pitch (rad), rate (rad/s)]  (2-dim, IMU-only; encoders ARE wired
                 but NOT yet fed into obs — the current policy is still 2-dim)
   Action      = 1 value u in [-1,1]          -> applied SIMULTANEOUSLY to both motors.

   4 changes that help the robot stay balanced (drawn from the stable L298N_package):
     1) STRONGER POLICY: policy_weights.h generated from example_policy_strong.onnx (kp=24,
        kd=1.0 rad). ACTUAL slope near the origin (from tanh nonlinearity) ~Kp 0.58/deg, ~Kd 0.025/deg/s,
        saturating around 1.7 deg -> a bit STIFFER + MORE damping than your cascade PID (0.49/0.014)
        => biased toward STABILITY (no oscillation). Previously the example was ~5x too weak -> the robot
        would drift to a large angle before reacting -> falls. (Want it softer: regenerate with --kp 20 --kd 0.6.)
     2) 200 Hz LOOP (was 60 Hz). The static PD policy (memoryless) can run fast SAFELY
        and more stably; it matches the PID's 200 Hz angle loop. (A REAL policy TRAINED
        in Isaac Lab MUST run at the frequency it was trained at -> set CONTROL_HZ back to 60.)
     3) SMART PWM SHAPER (MotorCommandShaper): below the breakaway threshold, use
        pulse-density (average torque ~ command) instead of jumping straight to MIN_DUTY (a
        torque step -> jitter/erratic movement). Also caps duty near balance + coasts on
        direction reversal. This is the main reason the robot couldn't "stand still" near balance before.
     4) STRONGER DUTY: MIN_DUTY 0.14, MAX_DUTY 0.90 (like the package) instead of 0.10/0.70.

   Network weights live in "policy_weights.h" (same folder), generated from the .onnx file:
     python pc_policy/export_policy_header.py --model pc_policy/example_policy_strong.onnx \
            --out firmware/rl_policy_esp32s3_l298n/policy_weights.h
   To change the policy = regenerate the header + reflash the firmware. Do not hand-edit the header.

   !!! CHECK DIRECTIONS ON FIRST RUN (wheels NOT touching the ground) — 2 steps:
   (1) ANTI-FALL DIRECTION (MOTOR_SIGN): tilt it forward by hand, both wheels must spin to
       "run toward the fall". Set MOTOR_SIGN=-1 (matches your PID sign). If wrong
       -> type 'm 1'.
   (2) BOTH WHEELS SAME DIRECTION (MOTOR_A_SIGN/MOTOR_B_SIGN): the 2 motors are usually
       mounted MIRRORED, so the same command spins them OPPOSITE -> the robot spins in place. Type 't' to
       test; if the wheels spin opposite -> type 'k -1' until they match.

   ARDUINO IDE: ESP32S3 Dev Module | USB CDC On Boot: Enabled | core >= 3.0
   WIRING (L298N + 2 encoders) — MATCHES the robot's ACTUAL pinout:
     I2C IMU : SDA=6  SCL=5            (!! SWAPPED vs old firmware SDA=5/SCL=6 - see README)
     Motor A : ENA=11 IN1=7  IN2=8     (LEFT wheel)
     Motor B : ENB=12 IN3=9  IN4=10    (RIGHT wheel)
     Encoder : left A=1 B=2 | right A=4 B=13  (wired; read for telemetry/future use,
               NOT yet fed into the 2-dim observation - the current policy is still IMU-only)
     LED RGB : 21 (onboard)
     L298N: feed motor power into VM(+12V/+VS), COMMON GND with the ESP32, logic 5V/GND.
            ENA/ENB take PWM (speed); INx take logic level (direction). Remove the ENA/ENB jumpers.
   SERIAL COMMANDS (115200):
     m 1/-1  flip anti-fall direction (MOTOR_SIGN)   g 1/-1  flip gyro sign
     j 1/-1  flip left wheel only (A)               k 1/-1  flip right wheel only (B)
     s -1.5  trim the balance angle (deg)           a 0.7   scale action (soften)
     n/l/b   MIN_DUTY / MAX_DUTY / deadband         c calib (HOLD UPRIGHT)
     t test 2 motors | x E-STOP | o re-enable       ?       print params
   LED: blue=calib | green=balancing | red=fallen/E-stop | red blink=IMU error
   ============================================================================= */

#include <Wire.h>
#include <math.h>
#include "policy_weights.h"

#if POLICY_OBS_DIM != 2
#error "This firmware builds a 2-dim obs [pitch, rate]. A model needing a different dim -> fix assembleObs() first."
#endif
#if POLICY_ACT_DIM != 1
#error "This firmware controls 1 axis (u) for both motors. A model with multiple actions -> fix the apply section."
#endif

// Struct Shaper is declared RIGHT AT THE TOP OF THE FILE: the Arduino IDE auto-generates a
// prototype for shaperUpdate(Shaper&) and inserts it at the top -> the 'Shaper' type MUST exist
// before that, otherwise you get "'Shaper' was not declared in this scope".
struct Shaper { float charge=0, remain=0, coast=0; int lastDir=0; };
Shaper shpL, shpR;

// =============================== USER CONFIG =================================
// ------- PINS (match the actual pinout) -------
static const int SDA_PIN = 6, SCL_PIN = 5, LED_PIN = 21;   // !! SDA/SCL swapped vs old firmware
// Motor A (LEFT wheel):  ENA = PWM (speed), IN1/IN2 = direction (logic)
static const int ENA = 11, IN1 = 7, IN2 = 8;
// Motor B (RIGHT wheel):  ENB = PWM (speed), IN3/IN4 = direction (logic)
static const int ENB = 12, IN3 = 9, IN4 = 10;
// Quadrature encoders (wired). Read for telemetry + ready for a future 4-dim obs;
// NOT fed into the current 2-dim policy (assembleObs below still uses IMU only).
static const int ENC_L_A = 1, ENC_L_B = 2;    // LEFT wheel encoder
static const int ENC_R_A = 4, ENC_R_B = 13;   // RIGHT wheel encoder
// Ticks/rev x1 (count rising edges on channel A). Package measured x4 CPR=1327 -> x1 ~ 332. Fix to the right value.
const float ENC_TICKS_PER_REV = 332.0f;

// ------- PWM (PWM on the L298N ENABLE pins) -------
static const int PWM_FREQ = 20000, PWM_RES = 10;   // same as L298N_package
static const uint32_t PWM_MAX_LEDC = (1u << PWM_RES) - 1u;

// ------- Motor mapping (u in [-1,1] -> duty), values taken from the well-working cascade PID -------
// deadband 0.04 (> the package's 0.03): swallows the MLP's ~0.03 static bias at balance -> no self-drift
float MIN_DUTY = 0.14f, MAX_DUTY = 0.90f, U_DEADBAND = 0.04f;
float LINEAR_START_U     = 0.10f;   // below this: ramp 0->MIN_DUTY + pulse-density
float NEAR_BAL_MAX_DUTY  = 0.42f;   // near balance: cap the duty (don't slam)
float NEAR_BAL_PITCH_DEG = 1.5f;    // "near balance" when |pitch|<= and |rate|<=
float NEAR_BAL_RATE_DPS  = 18.0f;
float U_SCALE     = 1.0f;           // <1 to run softer while testing ('a 0.5')
float PITCH_TRIM_DEG = 0.0f;        // trim the balance point (like the PID's targetDeg) - 's'
int   MOTOR_SIGN   = 1;            // ANTI-FALL direction (matches your PID sign)
int   MOTOR_A_SIGN = +1;            // flip LEFT wheel only  ('j -1') if mirror-mounted
int   MOTOR_B_SIGN = +1;            // flip RIGHT wheel only ('k -1') if mirror-mounted
int   GYRO_SIGN    = +1;

// ------- Timing / safety -------
// 200 Hz: the static PD policy runs fast -> more stable (matches the PID's 200Hz angle loop).
// A REAL policy TRAINED in sim (60Hz) -> set it back to 60: CONTROL_HZ = 60.
const int CONTROL_HZ = 100;
const unsigned long LOOP_US = 1000000UL / CONTROL_HZ;
const float MAX_SAFE_TILT_DEG = 45.0f;
const float COMP_ALPHA = 0.98f;
// PWM shaping (like MotorCommandShaper)
const float PULSE_WIDTH_S = 0.005f, REVERSE_COAST_S = 0.005f;
// ============================================================================

// ------- BMI160 -------
static const uint8_t REG_CHIP_ID=0x00, REG_DATA=0x0C, REG_ACC_CONF=0x40, REG_ACC_RANGE=0x41;
static const uint8_t REG_GYR_CONF=0x42, REG_GYR_RANGE=0x43, REG_CMD=0x7E;
static const uint8_t CMD_SOFT_RESET=0xB6, CMD_ACC_NORMAL=0x11, CMD_GYR_NORMAL=0x15;
static const float ACC_LSB_G=16384.0f, GYRO_LSB_DPS=131.2f;
uint8_t IMU_ADDR = 0x69;

// ------- state -------
float pitchDeg=0, rateDps=0, pitchFiltered=0, pitchOffsetDeg=0, gyroBiasY=0;
bool  filterSeeded=false, enabled=true;
float gAx,gAy,gAz,gGy;
float lastU=0, lastDutyFrac=0;
unsigned long lastLoopUs=0, inferUs=0;

// ------- encoder state (not used by the 2-dim policy; ready for a 4-dim one) -------
volatile int32_t encLCount=0, encRCount=0;   // tick counts (updated by ISR)
int32_t encLPrev=0, encRPrev=0;
float   wheelLVel=0, wheelRVel=0;             // wheel velocity (rad/s), from tick delta

// ============================ ENCODER (quadrature x1) ========================
void IRAM_ATTR isrEncL(){ if(digitalRead(ENC_L_B)) encLCount++; else encLCount--; }
void IRAM_ATTR isrEncR(){ if(digitalRead(ENC_R_B)) encRCount++; else encRCount--; }
void updateEncoders(float dt){
  int32_t l=encLCount, r=encRCount;           // 32-bit atomic read on the ESP32
  float k = (dt>1e-6f) ? (TWO_PI/ENC_TICKS_PER_REV/dt) : 0.0f;
  wheelLVel = (float)(l-encLPrev)*k;  encLPrev=l;
  wheelRVel = (float)(r-encRPrev)*k;  encRPrev=r;
}

// ============================ BMI160 I/O =====================================
void bmiWrite(uint8_t r,uint8_t v){ Wire.beginTransmission(IMU_ADDR); Wire.write(r); Wire.write(v); Wire.endTransmission(); }
bool bmiRead(uint8_t r,uint8_t*b,uint8_t n){
  Wire.beginTransmission(IMU_ADDR); Wire.write(r);
  if(Wire.endTransmission(false)!=0) return false;
  uint8_t g=Wire.requestFrom((int)IMU_ADDR,(int)n);
  for(uint8_t i=0;i<n && Wire.available();i++) b[i]=Wire.read();
  return g==n;
}
uint8_t bmiChipId(uint8_t a){
  Wire.beginTransmission(a); Wire.write(REG_CHIP_ID);
  if(Wire.endTransmission(false)!=0) return 0;
  if(Wire.requestFrom((int)a,1)!=1) return 0;
  return Wire.read();
}
bool bmiInit(){
  uint8_t cand[2]={0x69,0x68}; bool found=false;
  for(int i=0;i<2;i++){ if(bmiChipId(cand[i])==0xD1){ IMU_ADDR=cand[i]; found=true; break; } }
  if(!found){ for(int i=0;i<2;i++){ Wire.beginTransmission(cand[i]); if(Wire.endTransmission()==0){ IMU_ADDR=cand[i]; found=true; break; } } }
  if(!found) return false;
  bmiWrite(REG_CMD,CMD_SOFT_RESET); delay(100);
  bmiWrite(REG_CMD,CMD_ACC_NORMAL); delay(10);
  bmiWrite(REG_CMD,CMD_GYR_NORMAL); delay(100);
  bmiWrite(REG_ACC_RANGE,0x03); bmiWrite(REG_GYR_RANGE,0x03);
  bmiWrite(REG_ACC_CONF,0x2A);  bmiWrite(REG_GYR_CONF,0x2A);   // 400 Hz ODR (> 200Hz loop)
  delay(10);
  return true;
}
void readSensors(){
  uint8_t b[12]; if(!bmiRead(REG_DATA,b,12)) return;
  int16_t rgy=(int16_t)(((uint16_t)b[3]<<8)|b[2]);
  int16_t rax=(int16_t)(((uint16_t)b[7]<<8)|b[6]);
  int16_t ray=(int16_t)(((uint16_t)b[9]<<8)|b[8]);
  int16_t raz=(int16_t)(((uint16_t)b[11]<<8)|b[10]);
  gAx=rax/ACC_LSB_G; gAy=ray/ACC_LSB_G; gAz=raz/ACC_LSB_G;
  gGy=rgy/GYRO_LSB_DPS;
}

// ====================== ANGLE (like the package's PitchFilter) ================
void updateAngle(float dt){
  readSensors();
  float accA = atan2f(gAz, gAx) * RAD_TO_DEG;
  float rate = GYRO_SIGN * (gGy - gyroBiasY);
  if(!filterSeeded){ pitchFiltered=accA; filterSeeded=true; }
  else pitchFiltered = COMP_ALPHA*(pitchFiltered + rate*dt) + (1.0f-COMP_ALPHA)*accA;
  pitchDeg = pitchFiltered - pitchOffsetDeg;
  // >>> FIX: wrap to [-180,180]. With the sideways IMU mount (offset ~ -87 deg) an unwrapped angle
  // could read absurd values like 254 deg (see log), which broke the safety check and the policy input.
  while(pitchDeg >  180.0f) pitchDeg -= 360.0f;
  while(pitchDeg < -180.0f) pitchDeg += 360.0f;
  rateDps  = rate;
}
void calibrate(){
  Serial.println("# CALIB: GIU ROBOT DUNG THANG va YEN ~2s...");
  const int N=1000; double sA=0,sG=0;
  for(int i=0;i<N;i++){ readSensors(); sA+=atan2f(gAz,gAx)*RAD_TO_DEG; sG+=gGy; delay(2); }
  pitchOffsetDeg=(float)(sA/N); gyroBiasY=(float)(sG/N);
  filterSeeded=false; lastLoopUs=micros();
  Serial.printf("# CALIB offset=%.2f gyroBiasY=%.3f\n", pitchOffsetDeg, gyroBiasY);
}

// ===================== MLP FORWARD (replaces ONNX on the PC) ===================
static float bufA[POLICY_MAX_WIDTH], bufB[POLICY_MAX_WIDTH];

float policyForward(const float obs[POLICY_OBS_DIM]){
  float *in = bufA, *out = bufB;
  for(int i=0;i<POLICY_OBS_DIM;i++)
    in[i] = (obs[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];   // normalizer (1/1 if none)
  for(int l=0;l<POLICY_N_LAYERS;l++){
    const float *W = POLICY_W[l], *B = POLICY_B[l];
    const int nin = POLICY_LAYER_IN[l], nout = POLICY_LAYER_OUT[l];
    for(int j=0;j<nout;j++){
      const float *w = W + j*nin;
      float s = B[j];
      for(int i=0;i<nin;i++) s += w[i]*in[i];
      switch(POLICY_LAYER_ACT[l]){
        case ACT_ELU:     s = (s>0.0f) ? s : expf(s)-1.0f; break;
        case ACT_RELU:    if(s<0.0f) s=0.0f;               break;
        case ACT_TANH:    s = tanhf(s);                    break;
        case ACT_SIGMOID: s = 1.0f/(1.0f+expf(-s));        break;
      }
      out[j]=s;
    }
    float *t=in; in=out; out=t;
  }
  return in[0];
}

// Obs MUST match the training order + units: [pole_pos (rad), pole_vel (rad/s)].
// Subtract PITCH_TRIM_DEG so the policy aims at the true balance point (CoM not perfectly over the axle).
// When training a 4-dim env with encoders, set POLICY_OBS_DIM=4 (header from a 4-dim model) and:
//   cart_pos ~ wheel_radius * (left_wheel_angle + right_wheel_angle)/2   [m]
//   cart_vel ~ wheel_radius * (wheelLVel        + wheelRVel        )/2   [m/s]
//   obs[0]=pitch; obs[1]=rate; obs[2]=cart_pos; obs[3]=cart_vel;  (match the sim env's ORDER!)
void assembleObs(float obs[POLICY_OBS_DIM]){
  obs[0] = (pitchDeg - PITCH_TRIM_DEG) * DEG_TO_RAD;
  obs[1] = rateDps  * DEG_TO_RAD;
}

// ====================== MOTOR (L298N x2) + PWM shaping ======================
// Signal shaper, ported from L298N_package's MotorCommandShaper:
//  - below LINEAR_START_U: linear ramp 0->MIN_DUTY
//  - below MIN_DUTY: pulse-density (pulse MIN_DUTY on/off) -> average torque ~ command
//  - near balance: cap the duty (NEAR_BAL_MAX_DUTY)
//  - direction reversal: insert 1 coast frame (REVERSE_COAST_S) to avoid a current surge on reversal
// (struct Shaper + shpL/shpR declared AT THE TOP OF THE FILE - due to Arduino's prototype hoisting)

// returns signed duty in [-1,1] (sign = direction). near = currently near balance.
float shaperUpdate(Shaper &s, float signedU, bool near, float dt){
  dt = constrain(dt, 0.0001f, 0.020f);
  signedU = constrain(signedU, -1.0f, 1.0f);
  float absU = fabsf(signedU);
  float linStart = constrain(LINEAR_START_U, U_DEADBAND+0.001f, 1.0f);
  if(absU < U_DEADBAND){ s = Shaper{}; return 0.0f; }
  int reqDir = (signedU>0.0f) ? 1 : -1;
  if(s.lastDir!=0 && reqDir!=s.lastDir){            // direction reversal -> coast 1 frame
    s.charge=0; s.remain=0; s.coast=fmaxf(0.0f, REVERSE_COAST_S-dt); s.lastDir=reqDir; return 0.0f;
  }
  s.lastDir=reqDir;
  if(s.coast>0.0f){ s.coast=fmaxf(0.0f,s.coast-dt); return 0.0f; }
  float reqDuty;
  if(absU<linStart && MIN_DUTY>0.0f) reqDuty = MIN_DUTY*(absU/linStart);
  else { float nrm=(absU-linStart)/fmaxf(0.001f,1.0f-linStart);
         reqDuty = MIN_DUTY + constrain(nrm,0.0f,1.0f)*(MAX_DUTY-MIN_DUTY); }
  if(near) reqDuty = fminf(reqDuty, constrain(NEAR_BAL_MAX_DUTY, MIN_DUTY, MAX_DUTY));
  reqDuty = constrain(reqDuty, 0.0f, MAX_DUTY);
  float applied = reqDuty;
  if(MIN_DUTY>0.0f && reqDuty<MIN_DUTY){            // pulse-density below breakaway
    s.charge += (reqDuty/MIN_DUTY)*dt;
    if(s.remain<=0.0f && s.charge>=PULSE_WIDTH_S){ s.charge-=PULSE_WIDTH_S; s.remain=PULSE_WIDTH_S; }
    if(s.remain>0.0f){ applied=MIN_DUTY; s.remain=fmaxf(0.0f,s.remain-dt); }
    else applied=0.0f;
  } else { s.charge=0; s.remain=0; }
  if(applied<=0.0f) return 0.0f;
  return (reqDir>0) ? applied : -applied;
}

void writeMotorPins(int inA,int inB,int en,float signedDuty){
  if(signedDuty==0.0f){ ledcWrite(en,0); digitalWrite(inA,LOW); digitalWrite(inB,LOW); return; }
  bool fwd = signedDuty>0.0f;
  uint32_t d=(uint32_t)(fabsf(signedDuty)*PWM_MAX_LEDC+0.5f);
  digitalWrite(inA, fwd?HIGH:LOW); digitalWrite(inB, fwd?LOW:HIGH); ledcWrite(en,d);
}
void motorCoast(){
  lastDutyFrac=0; shpL=Shaper{}; shpR=Shaper{};
  ledcWrite(ENA,0); ledcWrite(ENB,0);
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
}
// u in [-1,1]: sign = anti-fall direction (MOTOR_SIGN). Applied to both motors via separate shapers.
void driveU(float u, float dt){
  float uc = constrain((float)MOTOR_SIGN*u, -1.0f, 1.0f);
  bool near = (fabsf(pitchDeg-PITCH_TRIM_DEG) <= NEAR_BAL_PITCH_DEG) &&
              (fabsf(rateDps) <= NEAR_BAL_RATE_DPS);
  float sdL = shaperUpdate(shpL, (float)MOTOR_A_SIGN*uc, near, dt);
  float sdR = shaperUpdate(shpR, (float)MOTOR_B_SIGN*uc, near, dt);
  writeMotorPins(IN1,IN2,ENA, sdL);
  writeMotorPins(IN3,IN4,ENB, sdR);
  lastDutyFrac = 0.5f*(fabsf(sdL)+fabsf(sdR));
}
void motorTest(){
  const uint32_t d=(uint32_t)(0.5f*PWM_MAX_LEDC);
  Serial.println("# TEST 2 banh THUAN 50% 1.5s (2 banh phai lan CUNG huong)...");
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW); ledcWrite(ENA,d);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW); ledcWrite(ENB,d);
  delay(1500); motorCoast(); delay(400);
  Serial.println("# TEST 2 banh NGHICH 50% 1.5s...");
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH); ledcWrite(ENA,d);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH); ledcWrite(ENB,d);
  delay(1500); motorCoast();
  Serial.println("# TEST xong. Neu 2 banh lan NGUOC nhau -> go 'k -1' (hoac 'j -1').");
  lastLoopUs=micros();
}

// ============================ SERIAL TUNER ===================================
void printParams(){
  Serial.printf("# PARAM scale=%.2f trim=%.2f MIN=%.2f MAX=%.2f dead=%.3f nearCap=%.2f "
                "MOTOR_SIGN=%d A=%d B=%d GYRO=%d en=%d | %dHz, policy %d lop obs=%d infer=%luus\n",
                U_SCALE, PITCH_TRIM_DEG, MIN_DUTY, MAX_DUTY, U_DEADBAND, NEAR_BAL_MAX_DUTY,
                MOTOR_SIGN, MOTOR_A_SIGN, MOTOR_B_SIGN, GYRO_SIGN, enabled,
                CONTROL_HZ, POLICY_N_LAYERS, POLICY_OBS_DIM, inferUs);
}
void parseCmd(char*s){
  char c=s[0]; float v=atof(s+1);
  switch(c){
    case 'a': U_SCALE=constrain(v,0.0f,1.0f); break;
    case 's': PITCH_TRIM_DEG=constrain(v,-20.0f,20.0f); break;
    case 'n': MIN_DUTY=constrain(v,0.0f,MAX_DUTY); break;      // clamp (avoid nonsense values via Serial)
    case 'l': MAX_DUTY=constrain(v,MIN_DUTY,1.0f); break;
    case 'b': U_DEADBAND=constrain(v,0.0f,0.95f); break;       // >1 would kill all motors -> block it
    case 'm': MOTOR_SIGN  =(v>=0)?+1:-1; break;
    case 'j': MOTOR_A_SIGN=(v>=0)?+1:-1; break;
    case 'k': MOTOR_B_SIGN=(v>=0)?+1:-1; break;
    case 'g': GYRO_SIGN   =(v>=0)?+1:-1; filterSeeded=false; break;
    case 'c': motorCoast(); calibrate(); break;
    case 't': motorCoast(); motorTest(); return;
    case 'x': enabled=false; motorCoast(); Serial.println("# STOP motor OFF"); break;
    case 'o': enabled=true; Serial.println("# GO motor ON"); break;
    case '?': default: printParams(); return;
  }
  printParams();
}
void handleSerial(){
  static char buf[32]; static uint8_t idx=0;
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n'||c=='\r'){ buf[idx]=0; if(idx>0) parseCmd(buf); idx=0; }
    else if(idx<sizeof(buf)-1) buf[idx++]=c;
  }
}
void telemetry(){
  static unsigned long last=0;
  if(millis()-last < 100) return;
  last=millis();
  Serial.printf("pitch=%6.2f rate=%7.1f u=%+.3f duty=%4.0f%% | encL=%ld encR=%ld velL=%5.1f velR=%5.1f | infer=%luus %s\n",
                pitchDeg, rateDps, lastU, lastDutyFrac*100.0f,
                (long)encLCount, (long)encRCount, wheelLVel, wheelRVel, inferUs,
                enabled ? "" : "[OFF]");
}

// ================================ SETUP ======================================
void setup(){
  Serial.begin(115200);
  // >>> FIX (standalone / no-USB): don't let Serial.printf block waiting for a USB host when the
  // board runs on battery / L298N-5V without a PC. This is code hardening only -- the "no LED when
  // USB unplugged" symptom is a POWER wiring issue (must feed 5V to the ESP32), NOT a code bug.
  Serial.setTxTimeoutMs(0);
  delay(500);
  neopixelWrite(LED_PIN,0,0,60);
  // L298N: ENABLE = PWM ; IN = direction logic
  ledcAttach(ENA,PWM_FREQ,PWM_RES);
  ledcAttach(ENB,PWM_FREQ,PWM_RES);
  pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);
  motorCoast();
  // Encoder: pulled-up inputs + interrupt on channel A (direction inferred from channel B)
  pinMode(ENC_L_A,INPUT_PULLUP); pinMode(ENC_L_B,INPUT_PULLUP);
  pinMode(ENC_R_A,INPUT_PULLUP); pinMode(ENC_R_B,INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrEncL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrEncR, RISING);
  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(400000);
  if(!bmiInit()){
    Serial.println("# LOI: khong thay BMI160 (kiem tra SDA=6/SCL=5, 3V3, GND)");
    while(true){ neopixelWrite(LED_PIN,80,0,0); delay(150); neopixelWrite(LED_PIN,0,0,0); delay(150); }
  }
  calibrate();

  // Self-test: run the network on a sample obs and time it — cross-check against
  // pc_policy (example_policy_strong.onnx: obs=[0.1,0] -> action = -0.9972).
  float testObs[POLICY_OBS_DIM] = {0.1f, 0.0f};
  unsigned long t0=micros();
  float testU = policyForward(testObs);
  inferUs = micros()-t0;
  Serial.printf("# SELF-TEST obs=[0.10, 0.00] -> action=%+.4f (infer %luus)\n", testU, inferUs);
  Serial.printf("# READY rl_policy L298N 2-motor %dHz. Go '?' xem lenh.\n", CONTROL_HZ);
  Serial.println("# KIEM TRA CHIEU truoc khi dat xuong: (1) 'm' chong nga, (2) 'j/k' 2 banh cung huong!");
  lastLoopUs=micros();
}

// ================================= LOOP ======================================
void loop(){
  handleSerial();

  unsigned long now=micros();
  if((now-lastLoopUs) < LOOP_US) return;
  float dt=(now-lastLoopUs)*1e-6f; lastLoopUs=now;

  updateAngle(dt);
  updateEncoders(dt);          // update wheel velocity (telemetry; not yet in obs)

  if(!enabled){ motorCoast(); neopixelWrite(LED_PIN,20,0,0); telemetry(); return; }
  if(fabsf(pitchDeg-PITCH_TRIM_DEG) > MAX_SAFE_TILT_DEG){    // fallen -> turn off both motors
    // >>> FIX: while fallen / being repositioned, re-seed the complementary filter to the
    // accelerometer so the pitch is correct IMMEDIATELY when the robot is set back upright.
    // Previously the filter took ~2 s to converge from a stale value, and during that time the
    // policy read a FALSE large tilt and lurched forward at full power (the "runs straight then
    // face-plants" symptom in the log: pitch falsely 37->8 deg while u was pinned at +1.0).
    filterSeeded=false;
    lastU=0; motorCoast(); neopixelWrite(LED_PIN,60,0,0); telemetry(); return;
  }

  float obs[POLICY_OBS_DIM];
  assembleObs(obs);
  unsigned long t0=micros();
  float u = policyForward(obs);
  inferUs = micros()-t0;

  u = constrain(u*U_SCALE, -1.0f, 1.0f);
  lastU=u;
  driveU(u, dt);
  neopixelWrite(LED_PIN,0,40,0);
  telemetry();
}

// ============================================================================
//  rl_policy TWIP  —  BAN NHE (MLP tay) HO TRO 1 HOAC 2 ACTION
//
//  Doc policy_weights.h (sinh tu export_policy_header.py). Tu dong theo so action:
//    - POLICY_ACT_DIM == 1 : 1 lenh u chung cho ca 2 banh (nhu truoc)
//    - POLICY_ACT_DIM == 2 : action[0]->banh A(trai), action[1]->banh B(phai)
//                            moi banh 1 lenh rieng, shaper rieng.
//
//  policy17: obs 2 [pitch,rate] -> 32(elu) -> 32(elu) -> 2(none)
//    (2 output = 2 banh. Neu thu tu nguoc: dung lenh 'w' de doi, hoac 'k -1'/'j -1'.)
//
//  Giu nguyen do hinh chay em: BMI160 tho, atan2(gAz,gAx)+offset, loc bu 0.98,
//  MotorCommandShaper (pulse-density), superloop 100Hz, serial tuner.
// ============================================================================
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "policy_weights.h"

#if POLICY_OBS_DIM != 2
#error "Firmware lap obs 2 chieu [pitch,rate]. Model khac chieu -> sua assembleObs()."
#endif
#if (POLICY_ACT_DIM != 1) && (POLICY_ACT_DIM != 2)
#error "Firmware ho tro 1 hoac 2 action. Model khac -> sua driveWheels()."
#endif

struct Shaper { float charge=0, remain=0, coast=0; int lastDir=0; };
Shaper shpL, shpR;

// =============================== USER CONFIG =================================
static const int SDA_PIN = 6, SCL_PIN = 5, LED_PIN = 21;
static const int ENA = 11, IN1 = 7, IN2 = 8;      // banh A (TRAI)
static const int ENB = 12, IN3 = 9, IN4 = 10;     // banh B (PHAI)
static const int ENC_L_A = 1, ENC_L_B = 2;
static const int ENC_R_A = 4, ENC_R_B = 13;
const float ENC_TICKS_PER_REV = 332.0f;

static const int PWM_FREQ = 20000, PWM_RES = 10;
static const uint32_t PWM_MAX_LEDC = (1u << PWM_RES) - 1u;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_A ENA
  #define PWM_B ENB
#else
  static const int CH_ENA = 0, CH_ENB = 1;
  #define PWM_A CH_ENA
  #define PWM_B CH_ENB
#endif

float MIN_DUTY = 0.14f, MAX_DUTY = 0.90f, U_DEADBAND = 0.04f;
float LINEAR_START_U     = 0.10f;
float NEAR_BAL_MAX_DUTY  = 0.42f;
float NEAR_BAL_PITCH_DEG = 1.5f;
float NEAR_BAL_RATE_DPS  = 18.0f;
float U_SCALE     = 1.0f;
float PITCH_TRIM_DEG = 0.0f;
int   MOTOR_SIGN   = 1;      // chieu CHONG NGA chung  ('m -1')
int   MOTOR_A_SIGN = +1;     // dao rieng banh A  ('j -1')
int   MOTOR_B_SIGN = +1;     // dao rieng banh B  ('k -1')
int   GYRO_SIGN    = +1;     // ('g -1')
int   WHEEL_SWAP   = 0;      // 1 = doi thu tu action[0]<->action[1]  ('w')  (chi tac dung khi 2 action)

const int CONTROL_HZ = 100;
const unsigned long LOOP_US = 1000000UL / CONTROL_HZ;
const float MAX_SAFE_TILT_DEG = 45.0f;
const float COMP_ALPHA = 0.98f;
const float PULSE_WIDTH_S = 0.005f, REVERSE_COAST_S = 0.005f;

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
float lastUL=0, lastUR=0, lastDutyFrac=0;
unsigned long lastLoopUs=0, inferUs=0;

volatile int32_t encLCount=0, encRCount=0;
int32_t encLPrev=0, encRPrev=0;
float   wheelLVel=0, wheelRVel=0;

// ===================== MLP FORWARD (obs -> POLICY_ACT_DIM output) =============
// Mang: (obs-mean)/std -> [FC + act]xN. Tra ra CA POLICY_ACT_DIM gia tri.
static float bufA[POLICY_MAX_WIDTH], bufB[POLICY_MAX_WIDTH];

void policyForward(const float obs[POLICY_OBS_DIM], float out[POLICY_ACT_DIM]){
  float *in = bufA, *o = bufB;
  for(int i=0;i<POLICY_OBS_DIM;i++)
    in[i] = (obs[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];   // chuan hoa (0/1 => giu nguyen)
  for(int l=0; l<POLICY_N_LAYERS; l++){
    const float *W = POLICY_W[l], *B = POLICY_B[l];
    const int nin = POLICY_LAYER_IN[l], nout = POLICY_LAYER_OUT[l];
    for(int j=0;j<nout;j++){
      const float *w = W + (long)j*nin;
      float s = B[j];
      for(int i=0;i<nin;i++) s += w[i]*in[i];
      switch(POLICY_LAYER_ACT[l]){
        case ACT_ELU:     s = (s>0.0f) ? s : expf(s)-1.0f; break;
        case ACT_RELU:    if(s<0.0f) s=0.0f;               break;
        case ACT_TANH:    s = tanhf(s);                    break;
        case ACT_SIGMOID: s = 1.0f/(1.0f+expf(-s));        break;
        // ACT_NONE: giu nguyen
      }
      o[j]=s;
    }
    float *t=in; in=o; o=t;
  }
  for(int i=0;i<POLICY_ACT_DIM;i++) out[i] = in[i];   // lay CA cac output
}

// ============================ ENCODER (quadrature x1) ========================
void IRAM_ATTR isrEncL(){ if(digitalRead(ENC_L_B)) encLCount++; else encLCount--; }
void IRAM_ATTR isrEncR(){ if(digitalRead(ENC_R_B)) encRCount++; else encRCount--; }
void updateEncoders(float dt){
  int32_t l=encLCount, r=encRCount;
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
  bmiWrite(REG_ACC_CONF,0x2A);  bmiWrite(REG_GYR_CONF,0x2A);
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

// ====================== ANGLE / OBS ==========================================
void updateAngle(float dt){
  readSensors();
  float accA = atan2f(gAz, gAx) * RAD_TO_DEG;
  float rate = GYRO_SIGN * (gGy - gyroBiasY);
  if(!filterSeeded){ pitchFiltered=accA; filterSeeded=true; }
  else pitchFiltered = COMP_ALPHA*(pitchFiltered + rate*dt) + (1.0f-COMP_ALPHA)*accA;
  pitchDeg = pitchFiltered - pitchOffsetDeg;
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
void assembleObs(float obs[POLICY_OBS_DIM]){
  obs[0] = (pitchDeg - PITCH_TRIM_DEG) * DEG_TO_RAD;
  obs[1] = rateDps  * DEG_TO_RAD;
}

// ====================== MOTOR SHAPER (pulse-density) =========================
float shaperUpdate(Shaper &s, float signedU, bool near, float dt){
  dt = constrain(dt, 0.0001f, 0.020f);
  signedU = constrain(signedU, -1.0f, 1.0f);
  float absU = fabsf(signedU);
  float linStart = constrain(LINEAR_START_U, U_DEADBAND+0.001f, 1.0f);
  if(absU < U_DEADBAND){ s = Shaper{}; return 0.0f; }
  int reqDir = (signedU>0.0f) ? 1 : -1;
  if(s.lastDir!=0 && reqDir!=s.lastDir){
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
  if(MIN_DUTY>0.0f && reqDuty<MIN_DUTY){
    s.charge += (reqDuty/MIN_DUTY)*dt;
    if(s.remain<=0.0f && s.charge>=PULSE_WIDTH_S){ s.charge-=PULSE_WIDTH_S; s.remain=PULSE_WIDTH_S; }
    if(s.remain>0.0f){ applied=MIN_DUTY; s.remain=fmaxf(0.0f,s.remain-dt); }
    else applied=0.0f;
  } else { s.charge=0; s.remain=0; }
  if(applied<=0.0f) return 0.0f;
  return (reqDir>0) ? applied : -applied;
}
void writeMotorPins(int inA,int inB,int pwmHandle,float signedDuty){
  if(signedDuty==0.0f){ ledcWrite(pwmHandle,0); digitalWrite(inA,LOW); digitalWrite(inB,LOW); return; }
  bool fwd = signedDuty>0.0f;
  uint32_t d=(uint32_t)(fabsf(signedDuty)*PWM_MAX_LEDC+0.5f);
  digitalWrite(inA, fwd?HIGH:LOW); digitalWrite(inB, fwd?LOW:HIGH); ledcWrite(pwmHandle,d);
}
void motorCoast(){
  lastDutyFrac=0; shpL=Shaper{}; shpR=Shaper{};
  ledcWrite(PWM_A,0); ledcWrite(PWM_B,0);
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
}
// Nhan lenh RIENG cho 2 banh (uL cho banh A, uR cho banh B). Da nhan U_SCALE truoc khi goi.
void driveWheels(float uL, float uR, float dt){
  float cmdL = constrain((float)MOTOR_SIGN*uL, -1.0f, 1.0f);
  float cmdR = constrain((float)MOTOR_SIGN*uR, -1.0f, 1.0f);
  bool near = (fabsf(pitchDeg-PITCH_TRIM_DEG) <= NEAR_BAL_PITCH_DEG) &&
              (fabsf(rateDps) <= NEAR_BAL_RATE_DPS);
  float sdL = shaperUpdate(shpL, (float)MOTOR_A_SIGN*cmdL, near, dt);
  float sdR = shaperUpdate(shpR, (float)MOTOR_B_SIGN*cmdR, near, dt);
  writeMotorPins(IN1,IN2,PWM_A, sdL);
  writeMotorPins(IN3,IN4,PWM_B, sdR);
  lastDutyFrac = 0.5f*(fabsf(sdL)+fabsf(sdR));
}
void motorTest(){
  const uint32_t d=(uint32_t)(0.5f*PWM_MAX_LEDC);
  Serial.println("# TEST 2 banh THUAN 50% 1.5s (phai lan CUNG huong)...");
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW); ledcWrite(PWM_A,d);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW); ledcWrite(PWM_B,d);
  delay(1500); motorCoast(); delay(400);
  Serial.println("# TEST 2 banh NGHICH 50% 1.5s...");
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH); ledcWrite(PWM_A,d);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH); ledcWrite(PWM_B,d);
  delay(1500); motorCoast();
  Serial.println("# TEST xong. 2 banh nguoc nhau -> 'k -1' (hoac 'j -1').");
  lastLoopUs=micros();
}

// ============================ SERIAL TUNER ===================================
void printParams(){
  Serial.printf("# PARAM scale=%.2f trim=%.2f MIN=%.2f MAX=%.2f dead=%.3f nearCap=%.2f "
                "MOTOR_SIGN=%d A=%d B=%d GYRO=%d swap=%d en=%d | %dHz, MLP %d lop obs=%d act=%d infer=%luus\n",
                U_SCALE, PITCH_TRIM_DEG, MIN_DUTY, MAX_DUTY, U_DEADBAND, NEAR_BAL_MAX_DUTY,
                MOTOR_SIGN, MOTOR_A_SIGN, MOTOR_B_SIGN, GYRO_SIGN, WHEEL_SWAP, enabled,
                CONTROL_HZ, POLICY_N_LAYERS, POLICY_OBS_DIM, POLICY_ACT_DIM, inferUs);
}
void parseCmd(char*s){
  char c=s[0]; float v=atof(s+1);
  switch(c){
    case 'a': U_SCALE=constrain(v,0.0f,1.0f); break;
    case 's': PITCH_TRIM_DEG=constrain(v,-20.0f,20.0f); break;
    case 'n': MIN_DUTY=constrain(v,0.0f,MAX_DUTY); break;
    case 'l': MAX_DUTY=constrain(v,MIN_DUTY,1.0f); break;
    case 'b': U_DEADBAND=constrain(v,0.0f,0.95f); break;
    case 'm': MOTOR_SIGN  =(v>=0)?+1:-1; break;
    case 'j': MOTOR_A_SIGN=(v>=0)?+1:-1; break;
    case 'k': MOTOR_B_SIGN=(v>=0)?+1:-1; break;
    case 'g': GYRO_SIGN   =(v>=0)?+1:-1; filterSeeded=false; break;
    case 'w': WHEEL_SWAP  =(v>=0.5f)?1:0; break;      // doi thu tu 2 action
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
  Serial.printf("pitch=%6.2f rate=%7.1f uL=%+.3f uR=%+.3f duty=%4.0f%% | encL=%ld encR=%ld velL=%5.1f velR=%5.1f | infer=%luus %s\n",
                pitchDeg, rateDps, lastUL, lastUR, lastDutyFrac*100.0f,
                (long)encLCount, (long)encRCount, wheelLVel, wheelRVel, inferUs,
                enabled ? "" : "[OFF]");
}

// ================================ SETUP ======================================
void setup(){
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(500);
  neopixelWrite(LED_PIN,0,0,60);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA, PWM_FREQ, PWM_RES);
  ledcAttach(ENB, PWM_FREQ, PWM_RES);
#else
  ledcSetup(CH_ENA, PWM_FREQ, PWM_RES); ledcAttachPin(ENA, CH_ENA);
  ledcSetup(CH_ENB, PWM_FREQ, PWM_RES); ledcAttachPin(ENB, CH_ENB);
#endif
  pinMode(IN1,OUTPUT); pinMode(IN2,OUTPUT);
  pinMode(IN3,OUTPUT); pinMode(IN4,OUTPUT);
  motorCoast();
  pinMode(ENC_L_A,INPUT_PULLUP); pinMode(ENC_L_B,INPUT_PULLUP);
  pinMode(ENC_R_A,INPUT_PULLUP); pinMode(ENC_R_B,INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrEncL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrEncR, RISING);

  Wire.begin(SDA_PIN,SCL_PIN); Wire.setClock(400000);
  if(!bmiInit()){
    Serial.println("# LOI: khong thay BMI160 (SDA=6/SCL=5, 3V3, GND)");
    while(true){ neopixelWrite(LED_PIN,80,0,0); delay(150); neopixelWrite(LED_PIN,0,0,0); delay(150); }
  }
  calibrate();

  // Self-test MLP: in CA POLICY_ACT_DIM output cho 2 obs mau (so voi exporter --check)
  {
    float t1[POLICY_OBS_DIM]={0.1f,0.0f}, o1[POLICY_ACT_DIM];
    unsigned long t0=micros(); policyForward(t1,o1); inferUs=micros()-t0;
    float t2[POLICY_OBS_DIM]={-0.5f,0.5f}, o2[POLICY_ACT_DIM];
    policyForward(t2,o2);
    Serial.print("# SELF-TEST [0.1,0]->[");
    for(int i=0;i<POLICY_ACT_DIM;i++) Serial.printf("%+.4f%s", o1[i], i<POLICY_ACT_DIM-1?", ":"");
    Serial.print("]  [-0.5,0.5]->[");
    for(int i=0;i<POLICY_ACT_DIM;i++) Serial.printf("%+.4f%s", o2[i], i<POLICY_ACT_DIM-1?", ":"");
    Serial.printf("]  infer=%luus (so voi export --check)\n", inferUs);
  }
  Serial.printf("# READY MLP %d lop, %d action, %dHz. Go '?' xem lenh.\n",
                POLICY_N_LAYERS, POLICY_ACT_DIM, CONTROL_HZ);
  Serial.println("# KIEM CHIEU: (1) 'm' chong nga, (2) 'j/k' 2 banh cung huong, (3) neu xoay -> 'w 1'.");
  lastLoopUs=micros();
}

// ================================= LOOP ======================================
void loop(){
  handleSerial();
  unsigned long now=micros();
  if((now-lastLoopUs) < LOOP_US) return;
  float dt=(now-lastLoopUs)*1e-6f; lastLoopUs=now;

  updateAngle(dt);
  updateEncoders(dt);

  if(!enabled){ motorCoast(); neopixelWrite(LED_PIN,20,0,0); telemetry(); return; }
  if(fabsf(pitchDeg-PITCH_TRIM_DEG) > MAX_SAFE_TILT_DEG){
    filterSeeded=false; lastUL=lastUR=0; motorCoast(); neopixelWrite(LED_PIN,60,0,0); telemetry(); return;
  }

  float obs[POLICY_OBS_DIM]; assembleObs(obs);
  float act[POLICY_ACT_DIM];
  unsigned long t0=micros();
  policyForward(obs, act);              // <-- tra ra POLICY_ACT_DIM gia tri
  inferUs = micros()-t0;

#if POLICY_ACT_DIM == 1
  float uL = act[0], uR = act[0];       // 1 lenh chung cho 2 banh
#else
  float uL, uR;                         // 2 lenh: moi banh 1
  if(WHEEL_SWAP){ uL = act[1]; uR = act[0]; }
  else          { uL = act[0]; uR = act[1]; }
#endif
  uL = constrain(uL*U_SCALE, -1.0f, 1.0f);
  uR = constrain(uR*U_SCALE, -1.0f, 1.0f);
  lastUL=uL; lastUR=uR;

  driveWheels(uL, uR, dt);
  neopixelWrite(LED_PIN,0,40,0);
  telemetry();
}
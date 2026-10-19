// Landing detector: initial contact -> maximum knee flexion, from the hip and ankle
// keypoint trajectories. Same logic and thresholds as scripts/suggest_frames.py
// (tuned on the blind pilot), run online one frame at a time.

export const FALL_SPEED = 0.015;   // feet falling if moving down faster than this x leg length per frame (30 fps)
export const STOP_SPEED = 0.010;   // feet stopped if moving down slower than this
export const FLEX_SEARCH_S = 0.6;  // max flexion must come within this many seconds of contact
const RISE_FRAMES = 2;             // hip rising for this many frames -> max flexion has passed
const VIS_MIN = 0.5;
// walking-in guards: a person entering the view or walking towards the camera also moves both feet down the image
const SETTLE_S = 0.7;              // legs must be in view this long before a landing can start
const LOST_S = 0.3;                // legs out of view longer than this = the person left
const WALK_SIDE = 0.03;            // hips moving sideways faster than this x leg length per frame = walking
const WALK_HOLD_S = 0.4;           // after walking is seen, ignore falls for this long
const FLOOR_GAP_S = 0.7;           // floor level = lowest feet position seen, ignoring the last 0.7 s ...
const FLOOR_S = 30;                // ... over the last 30 s
const FLOOR_TOL = 0.05;            // a landing may end at most this x leg length below that floor: walking towards
                                   // the camera keeps moving the feet lower down the image, a jump comes back to it

const L_HIP = 23, R_HIP = 24, L_ANKLE = 27, R_ANKLE = 28;

export class LandingDetector {
  constructor(onLanding, onTakeoff) {
    this.onLanding = onLanding;    // called with {contact, flex} frame numbers
    this.onTakeoff = onTakeoff;    // called with the time (s) of the next take-off after a landing
    this.prev = null;              // {frame, time, ankleY, ...}
    this.legs = [];
    this.falling = false;
    this.landing = null;           // {contact, contactTime, maxHip, maxFrame, rising}
    this.awaitTakeoff = false;     // a landing was reported and its next take-off is not seen yet
    this.visibleSince = null;      // time the legs came (back) into view
    this.lostSince = null;         // time the legs went out of view
    this.walkUntil = -1;           // falls before this time are ignored (walking seen)
    this.feet = [];                // recent {t, y} of the lower foot, for the floor level
  }

  // lm: 33 BlazePose landmarks (normalised), w/h: frame size in px
  push(frame, time, lm, w, h) {
    const vis = [L_HIP, R_HIP, L_ANKLE, R_ANKLE].every(i => (lm[i].visibility ?? 1) >= VIS_MIN);
    if (!vis) {
      // legs lost from view: drop any half-finished landing instead of waiting for the legs to come back
      this.prev = null; this.falling = false; this.landing = null;
      if (this.lostSince === null) this.lostSince = time;
      if (time - this.lostSince > LOST_S) {      // really gone, not a one-frame glitch: a new person may walk in
        this.visibleSince = null; this.legs = []; this.feet = [];
      }
      return;
    }
    this.lostSince = null;
    if (this.visibleSince === null) this.visibleSince = time;
    const leftY = lm[L_ANKLE].y * h, rightY = lm[R_ANKLE].y * h;
    const ankleY = (leftY + rightY) / 2;
    const hipY = (lm[L_HIP].y + lm[R_HIP].y) / 2 * h;
    const hipX = (lm[L_HIP].x + lm[R_HIP].x) / 2 * w;
    this.legs.push(Math.abs(ankleY - hipY));
    if (this.legs.length > 60) this.legs.shift();
    this.feet.push({ t: time, y: Math.max(leftY, rightY) });
    while (time - this.feet[0].t > FLOOR_S) this.feet.shift();
    const leg = median(this.legs);

    if (this.landing) {                      // looking for maximum knee flexion = lowest hip
      const L = this.landing;
      if (hipY > L.maxHip) { L.maxHip = hipY; L.maxFrame = frame; L.rising = 0; }
      else if (hipY < L.maxHip - 0.005 * leg) L.rising++;
      if (L.rising >= RISE_FRAMES || time - L.contactTime > FLEX_SEARCH_S) {
        const flex = Math.max(L.maxFrame, L.contact + 1);
        this.landing = null;
        this.awaitTakeoff = true;
        this.onLanding({ contact: L.contact, flex, contactTime: L.contactTime });
      }
    } else if (this.prev && leg > 0) {       // looking for initial contact
      const dt = Math.max(1e-3, time - this.prev.time);
      const k = 1 / leg * (1 / 30) / dt;                       // per 30-fps frame, in leg lengths
      const speed = (ankleY - this.prev.ankleY) * k;            // + = down
      const speedL = (leftY - this.prev.leftY) * k, speedR = (rightY - this.prev.rightY) * k;
      if (this.awaitTakeoff && speed < -FALL_SPEED) {   // feet moving up fast again: next take-off
        this.awaitTakeoff = false;
        this.onTakeoff?.(time);
      }
      // walking across the view: ignore falls for a short time
      const side = Math.abs(hipX - this.prev.hipX) * k;
      this.sideAvg = 0.6 * (this.sideAvg ?? side) + 0.4 * side;
      if (this.sideAvg > WALK_SIDE) this.walkUntil = time + WALK_HOLD_S;
      const settled = time - this.visibleSince >= SETTLE_S && time >= this.walkUntil;
      // a jump lands on both feet: both ankles must fall together (walking moves one foot at a time)
      if (!settled) this.falling = false;
      else if (Math.min(speedL, speedR) > FALL_SPEED) this.falling = true;
      else if (this.falling && speed < STOP_SPEED) {
        this.falling = false;
        if (this.onFloor(time, leg))         // not a new lower point = not walking towards the camera
          this.landing = { contact: frame, contactTime: time, maxHip: hipY, maxFrame: frame, rising: 0 };
      }
    }
    this.prev = { frame, time, ankleY, leftY, rightY, hipX };
  }

  // the feet came down to (or above) a floor level already seen, not to a new lower point in the image
  onFloor(time, leg) {
    let floor = -Infinity;
    for (const f of this.feet) if (time - f.t >= FLOOR_GAP_S && f.y > floor) floor = f.y;
    return floor > -Infinity && this.feet[this.feet.length - 1].y <= floor + FLOOR_TOL * leg;
  }
}

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

// ---- angles (same definitions as the Python helper) ----
const L_SH = 11, R_SH = 12, L_KNEE = 25, R_KNEE = 26;

function fppa(hip, knee, ankle, midHip) {
  const thigh = [knee[0] - hip[0], knee[1] - hip[1]], shank = [ankle[0] - knee[0], ankle[1] - knee[1]];
  const cos = (thigh[0] * shank[0] + thigh[1] * shank[1]) /
    (Math.hypot(...thigh) * Math.hypot(...shank) + 1e-9);
  const angle = Math.acos(Math.min(1, Math.max(-1, cos))) * 180 / Math.PI;
  const line = [ankle[0] - hip[0], ankle[1] - hip[1]];
  const side = p => Math.sign(line[0] * (p[1] - hip[1]) - line[1] * (p[0] - hip[0]));
  return side(knee) === side(midHip) ? angle : -angle;
}

export function frameAngles(lm, w, h) {
  const keys = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE];
  if (!keys.every(i => (lm[i].visibility ?? 1) >= VIS_MIN)) return null;
  const P = i => [lm[i].x * w, lm[i].y * h];
  const midHip = [(P(L_HIP)[0] + P(R_HIP)[0]) / 2, (P(L_HIP)[1] + P(R_HIP)[1]) / 2];
  const midSh = [(P(L_SH)[0] + P(R_SH)[0]) / 2, (P(L_SH)[1] + P(R_SH)[1]) / 2];
  const v = [midSh[0] - midHip[0], midSh[1] - midHip[1]];
  return {
    left: fppa(P(L_HIP), P(L_KNEE), P(L_ANKLE), midHip),
    right: fppa(P(R_HIP), P(R_KNEE), P(R_ANKLE), midHip),
    trunk: Math.atan2(Math.abs(v[0]), -v[1]) * 180 / Math.PI,
  };
}

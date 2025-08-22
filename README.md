# tello_autonomous_navigation
a apriltag detection and color for fira air U19 competition 

from djitellopy import tello
import cv2
import numpy as np
import time
import logging
import threading
import sys
import at

TAG_LIST = [1,2,3,4,5]
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
CX = FRAME_WIDTH // 2
CY = FRAME_HEIGHT // 2

# ------------------------------VARIABLES------------------------------
#target distance
TARGET_DIST_CM = 60
TOL_XY_PX = 12
TOL_YAW_DEG = 8         
TOL_DIST_CM = 10           
TOL_H_ALIGN_PX = 14     
DESCENT_STEP_CM = 18    
FORWARD_PASS_CM = 110     
GATE_ALIGN_TOL_PX = 18   

#safety
LOW_BATTEY_LAND = 15
NO_TAG_SEEN_TIMEOUT = 4.0
NO_H_SEEN_TIMEOUT = 5.0
MAX_RC = 40
#search
SEARCH_YAW_DEPS = 28
SEARCH_VZ = 26
SEARCH_MIN_H = 60     #########################################################
SEARCH_MAX_H = 150    ############################################################

#gate color ranges(HSV):
GATE_COLOR_DB = {
    "blue":  ([90, 80, 80],  [130, 255, 255]),
    "green": ([40, 60, 60],  [75, 255, 255]),
    "red1":  ([0, 90, 90],   [10, 255, 255]),
    "red2":  ([170, 90, 90], [180, 255, 255]),
    "orange":([11, 120, 120],[25, 255, 255]),
    "purple":([130, 60, 60], [155, 255, 255]),
}

# ============================================================
                             #PID
# ============================================================
KPX =0.48
KIX =0.0009
KDX =  0.22
KPY = 0.50 
KIY = 0.0010
KDY = 0.22
KPD = 0.55 
KID = 0.0012
KDD =  0.26  
KPYAW = 0.62
KIYAW =0.0009 
KDYAW =  0.28

#prev errors/ integrals[i]
int_x = 0.0
int_y = 0.0
int_d = 0.0
int_yaw = 0.0

prev_x = 0.0
prev_y = 0.0
prev_d = 0.0
prev_yaw = 0.0

prev_pid_time = time.time()

#EMA smooting for raw error
ema_x=0.0
ema_y = 0.0
ema_d = 0.0
ema_yaw = 0.0
EMA_ALPHA = 0.25

# cv2.setUseOptimized(True)
# cv2.setNumThreads(0)

def ema_update(prev, new, alpha=EMA_ALPHA):
    return (1.0 - alpha) * prev + alpha * float(new)

def pid_compute(err, kp, ki, kd, integ, prev_err, dt, out_limit=MAX_RC, integ_limit=2000.0):
    # Anti-windup integrator clamp
    integ = np.clip(integ + err * dt, -integ_limit, integ_limit)
    deriv = (err - prev_err) / dt if dt > 1e-4 else 0.0
    out = kp * err + ki * integ + kd * deriv
    out = int(np.clip(out, -out_limit, out_limit))
    return out, integ, err

def pid_step(ex, ey, ed, eyaw):
    global int_x, int_y, int_d, int_yaw
    global prev_x, prev_y, prev_d, prev_yaw, prev_pid_time
    # time delta
    now = time.time()
    dt = max(1e-3, now - prev_pid_time)
    prev_pid_time = now

    # compute each axis
    vx, int_x, prev_x  = pid_compute(ex,   KPX,   KIX,   KDX,   int_x,   prev_x,   dt)
    vy, int_y, prev_y  = pid_compute(ey,   KPY,   KIY,   KDY,   int_y,   prev_y,   dt)
    vd, int_d, prev_d  = pid_compute(ed,   KPD,   KID,   KDD,   int_d,   prev_d,   dt)
    vyaw, int_yaw, prev_yaw = pid_compute(eyaw, KPYAW, KIYAW, KDYAW, int_yaw, prev_yaw, dt)

    return vx, vy, vd, vyaw


def generate_h_template(width=90, height=90, thickness=14, gap=18):
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(canvas, (int(width*0.15), int(height*0.15)),
                          (int(width*0.15)+thickness, int(height*0.85)), 255, -1)
    cv2.rectangle(canvas, (int(width*0.85)-thickness, int(height*0.15)),
                          (int(width*0.85), int(height*0.85)), 255, -1)
    
    cy = height // 2
    cv2.rectangle(canvas, (int(width*0.15), cy - gap//2),
                          (int(width*0.85), cy + gap//2), 255, -1)


    canvas = cv2.GaussianBlur(canvas, (3,3), 0.8)
    return canvas

H_TEMPLATE = generate_h_template(110, 110, thickness=16, gap=20)


def detect_gate(frame_bgr):
    """
    Multi-color gate detector using HSV masks + morphology.
    Returns (center_x, center_y), (w,h) of the largest plausible gate blob, or (None, None)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    for name, (lo, hi) in GATE_COLOR_DB.items():
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        if name == "red1" or name == "red2":
            # handled individually below to union both red ranges
            pass
        else:
            mask = cv2.inRange(hsv, lo, hi)
            mask_total = cv2.bitwise_or(mask_total, mask)

    # Red union
    lo1 = np.array(GATE_COLOR_DB["red1"][0], dtype=np.uint8)
    hi1 = np.array(GATE_COLOR_DB["red1"][1], dtype=np.uint8)
    lo2 = np.array(GATE_COLOR_DB["red2"][0], dtype=np.uint8)
    hi2 = np.array(GATE_COLOR_DB["red2"][1], dtype=np.uint8)
    mask_r1 = cv2.inRange(hsv, lo1, hi1)
    mask_r2 = cv2.inRange(hsv, lo2, hi2)
    mask_total = cv2.bitwise_or(mask_total, cv2.bitwise_or(mask_r1, mask_r2))

    # clean up
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask_total = cv2.medianBlur(mask_total, 5)

    cnts, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 900:  # reject tiny blobs
        return None, None

    x,y,w,h = cv2.boundingRect(c)
    cx, cy = x + w//2, y + h//2
    cv2.rectangle(frame_bgr, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.circle(frame_bgr, (cx,cy), 6, (255,0,0), -1)
    return (cx,cy), (w,h)

# ============================================================
#                Vision: H Detection (robust)
# ============================================================

def detect_H(frame_bgr, template=H_TEMPLATE):
    """
    Hybrid approach:
    1) Adaptive threshold + morphology to emphasize shapes
    2) Template matching (TM_CCOEFF_NORMED)
    3) Optional contour heuristic to confirm H-like geometry
    Returns center (cx, cy) if found, else None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # adaptive binarization
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 9)
    # clean noise
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Template matching on binarized frame
    res = cv2.matchTemplate(bin_img, template, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minl, maxl = cv2.minMaxLoc(res)

    if maxv < 0.68:  # stricter threshold to reduce false-positives
        return None

    th, tw = template.shape[:2]
    top_left = maxl
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cx = top_left[0] + tw // 2
    cy = top_left[1] + th // 2

    # Extra geometry heuristic: aspect within plausible H bounding box
    roi = bin_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if roi.size == 0:
        return None
    # Check existence of three vertical/horizontal clusters using projection profile
    col_sum = np.sum(roi, axis=0) / 255.0
    row_sum = np.sum(roi, axis=1) / 255.0
    if np.max(col_sum) < 0.3 * roi.shape[0] or np.max(row_sum) < 0.15 * roi.shape[1]:
        # too weak structure
        return None

    cv2.rectangle(frame_bgr, top_left, bottom_right, (0,255,0), 2)
    cv2.circle(frame_bgr, (cx,cy), 6, (0,255,0), -1)
    return (cx, cy)

# ============================================================
#                AprilTag wrapper (from at)
# ============================================================

def find_current_tag(tags, desired_id):
    """
    tags: list of tuples (id, err_x, err_y, dist, yaw)
    desired_id: id to find
    returns tuple or None
    """
    for t in tags:
        if int(t[0]) == int(desired_id):
            return t
    return None

# ============================================================
#                   Search Pattern (strong)
# ============================================================

search_dir = 1       # +1 going up, -1 going down
last_search_flip = time.time()

def search_step(current_height_cm):
    """
    Strong search: continuous yaw + vertical sweep between SEARCH_MIN_H..SEARCH_MAX_H
    Returns (vx, vd, vz, vyaw) rc commands limited to MAX_RC
    """
    global search_dir, last_search_flip

    # adjust direction if height bounds reached
    if current_height_cm is not None:
        if current_height_cm > SEARCH_MAX_H:
            search_dir = -1
            last_search_flip = time.time()
        elif current_height_cm < SEARCH_MIN_H:
            search_dir = +1
            last_search_flip = time.time()

    vz = int(np.clip(SEARCH_VZ * search_dir, -MAX_RC, MAX_RC))
    vyaw = int(np.clip(SEARCH_YAW_DEPS, -MAX_RC, MAX_RC))
    vx = 0
    vd = 0
    return vx, vd, vz, vyaw

# ============================================================
#                Tello Setup & RC loop
# ============================================================

bot = tello.Tello()
bot.LOGGER.setLevel(logging.ERROR)

def tello_connect_or_die():
    try:
        bot.connect()
        print(f" Connected. Battery: {bot.get_battery()}%")
    except Exception as e:
        print(f" Cannot connect to Tello: {e}")
        sys.exit(1)

tello_connect_or_die()

# Stream
try:
    bot.streamon()
except Exception as e:
    print(f" Cannot start stream: {e}")
    bot.end()
    sys.exit(1)

reader = bot.get_frame_read(with_queue=False)

# Takeoff
try:
    bot.takeoff()
except Exception as e:
    print(f" Takeoff failed: {e}")
    bot.end()
    sys.exit(1)

time.sleep(2.5)

# RC command variables (updated by main loop, sent by thread)
rc_vx = 0    # left/right (+right)
rc_vd = 0    # forward/back (+forward)
rc_vz = 0    # up/down (+up)
rc_vyaw = 0  # yaw (+ccw)

running = True

def rc_thread():
    global rc_vx, rc_vd, rc_vz, rc_vyaw, running
    while running:
        try:
            bot.send_rc_control(int(rc_vx), int(rc_vd), int(rc_vz), int(rc_vyaw))
        except Exception as e:
            print(f"⚠️ RC send failed: {e}")
            try:
                bot.send_rc_control(0,0,0,0)
            except: pass
        time.sleep(0.05)  # 20Hz

thr = threading.Thread(target=rc_thread, daemon=True)
thr.start()

# ============================================================
#                    State Machine
# ============================================================

STATE = "SEARCH_TAGS"  # SEARCH_TAGS -> TRACK_TAG -> PASS_GATE(opt) -> NEXT_TAG or FIND_H -> LANDING
tag_index = 0
last_seen_time = time.time()
last_h_seen = 0.0

# Move-average error memory (for smoother PID)
ema_x = ema_y = ema_d = ema_yaw = 0.0

def reset_pid():
    global int_x, int_y, int_d, int_yaw
    global prev_x, prev_y, prev_d, prev_yaw
    int_x = int_y = int_d = int_yaw = 0.0
    prev_x = prev_y = prev_d = prev_yaw = 0.0

# ============================================================
#                         MAIN LOOP
# ============================================================

try:
    while True:
        # --- Read frame safely ---
        frame_rgb = reader.frame  # RGB
        if frame_rgb is None:
            print(" Empty frame; keeping hover...")
            rc_vx = rc_vd = rc_vz = rc_vyaw = 0
            continue

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))
        vis = frame_bgr.copy()

        # Telemetry
        try:
            height_cm = bot.get_height()
        except:
            height_cm = None

        battery = bot.get_battery()
        if battery is not None and battery < LOW_BATTEY_LAND:
            print(" Low battery; landing safely.")
            rc_vx = rc_vd = rc_vz = rc_vyaw = 0
            bot.land()
            break

        # Detect AprilTags
        try:
            tags = at.get_tags(frame_bgr)  
        except Exception as e:
            tags = []
            cv2.putText(vis, f"AT error: {str(e)}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Detect Gate (color)
        gate_center, gate_size = detect_gate(vis)

        # Detect H (only in FIND_H/LANDING states or opportunistically to show UI)
        h_center_draw = detect_H(vis) if STATE in ("FIND_H","LANDING") else None

        # --------------- State Logic ---------------

        # ============== SEARCH_TAGS: spin + vertical sweep until current tag is seen
        if STATE == "SEARCH_TAGS":
            # Find the current tag if exists
            if tag_index < len(TAG_LIST):
                desired_id = TAG_LIST[tag_index]
                found = find_current_tag(tags, desired_id)
                if found is not None:
                    last_seen_time = time.time()
                    STATE = "TRACK_TAG"
                    reset_pid()
                    print(f" TAG {desired_id} found -> TRACK_TAG")
                    # small brake
                    rc_vx = rc_vd = rc_vz = 0
                    rc_vyaw = 0
                else:
                    # robust search step
                    vx, vd, vz, vyaw = search_step(height_cm)
                    rc_vx, rc_vd, rc_vz, rc_vyaw = vx, vd, vz, vyaw
            else:
                # No more tags -> find H
                print(" All TAGs done -> FIND_H")
                STATE = "FIND_H"
                reset_pid()
                rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                last_h_seen = 0.0

        # ============== TRACK_TAG: PID towards tag; pass through center/gate when aligned
        elif STATE == "TRACK_TAG":
            desired_id = TAG_LIST[tag_index]
            t = find_current_tag(tags, desired_id)

            if t is None:
                # lost tag -> if too long, search again
                if (time.time() - last_seen_time) > NO_TAG_SEEN_TIMEOUT:
                    print(" Lost tag -> SEARCH_TAGS")
                    STATE = "SEARCH_TAGS"
                    reset_pid()
                    rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                else:
                    # hover lightly and rotate a bit
                    rc_vx = rc_vd = rc_vz = 0
                    rc_vyaw = 18
            else:
                # Extract and smooth errors
                tid, ex, ey, dist, yaw = t
                ex = float(ex)
                ey = float(-ey)  # invert to align with RC z orientation (up positive)
                ed = float(dist - TARGET_DIST_CM)
                eyaw_e = float(yaw)

                # EMA smoothing
                ema_x = ema_update(ema_x, ex)
                ema_y = ema_update(ema_y, ey)
                ema_d = ema_update(ema_d, ed)
                ema_yaw = ema_update(ema_yaw, eyaw_e)

                # PID
                vx, vz, vd, vyaw = pid_step(ema_x, ema_y, ema_d, ema_yaw)

                rc_vx = vx
                rc_vz = vz
                rc_vd = vd
                rc_vyaw = vyaw

                last_seen_time = time.time()

                # Show UI
                cv2.putText(vis, f"TAG {desired_id}  x:{int(ex)} y:{int(-ey)} d:{int(dist)} yaw:{int(yaw)}",
                            (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.circle(vis, (CX, CY), 8, (0,0,255), -1)

                # Condition: centered & within distance & yaw tolerance -> forward pass
                if (abs(ex) < GATE_ALIGN_TOL_PX and abs(ey) < GATE_ALIGN_TOL_PX and
                    abs(yaw) < TOL_YAW_DEG and abs(dist - TARGET_DIST_CM) < TOL_DIST_CM):
                    print("✅ Centered on TAG -> Forward pass through gate/marker")
                    # brake then pass
                    rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                    time.sleep(0.2)
                    try:
                        bot.move_forward(FORWARD_PASS_CM)
                    except Exception as e:
                        print(f"⚠️ move_forward failed: {e}")
                    # tag done -> next
                    tag_index += 1
                    reset_pid()
                    rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                    if tag_index >= len(TAG_LIST):
                        print("✅ All TAGs done -> FIND_H")
                        STATE = "FIND_H"
                    else:
                        print(f"➡️ Next TAG -> SEARCH_TAGS")
                        STATE = "SEARCH_TAGS"

        # ============== FIND_H: search for H; when found, align & descend to land
        elif STATE == "FIND_H":
            # try detect H
            h_center = detect_H(vis)
            if h_center is not None:
                last_h_seen = time.time()
                hx, hy = h_center

                ex = float(hx - CX)
                ey = float(-(hy - CY))

                # Only x,y alignment for H (no forward nor yaw needed typically)
                vx, vz, _, vyaw = pid_step(ex, ey, 0.0, 0.0)

                # Keep yaw slowly rotating to stabilize (optional)
                vyaw = int(np.clip(vyaw, -12, 12))

                rc_vx = vx
                rc_vz = vz
                rc_vd = 0
                rc_vyaw = vyaw

                cv2.putText(vis, "H DETECTED - Aligning...", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # If centered, start landing phase
                if abs(ex) < TOL_H_ALIGN_PX and abs(ey) < TOL_H_ALIGN_PX:
                    print("🎯 Centered on H -> LANDING")
                    STATE = "LANDING"
                    rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                    time.sleep(0.2)
            else:
                # Not seeing H -> search spin with gentle vertical dithering
                dt_no_h = time.time() - last_h_seen if last_h_seen > 0 else 999
                # vertical oscillation to catch different scales
                vz = int(18 * np.sign(np.sin(time.time()*0.8)))
                rc_vx = 0
                rc_vd = 0
                rc_vyaw = 24
                rc_vz = vz

                if dt_no_h > NO_H_SEEN_TIMEOUT and height_cm is not None:
                    # widen search by changing altitude band
                    print("🔎 Expanding H search band...")
                    # small nudge up or down
                    rc_vz = 26 if height_cm < (SEARCH_MIN_H+SEARCH_MAX_H)//2 else -26

        # ============== LANDING: final descent while keeping centered on H
        elif STATE == "LANDING":
            h_center = detect_H(vis)
            if h_center is not None:
                hx, hy = h_center
                ex = float(hx - CX)
                ey = float(-(hy - CY))
                vx, vz, _, vyaw = pid_step(ex, ey, 0.0, 0.0)
                # During landing, keep movements gentle
                rc_vx = int(np.clip(vx, -22, 22))
                rc_vz = int(np.clip(vz, -22, 22))
                rc_vyaw = int(np.clip(vyaw, -10, 10))
                rc_vd = 0

                # If centered tightly, descend a bit, else hold altitude correction
                if abs(ex) < TOL_H_ALIGN_PX and abs(ey) < TOL_H_ALIGN_PX:
                    # brief brake
                    rc_vx = rc_vd = rc_vyaw = 0
                    rc_vz = 0
                    try:
                        print("⬇️ Descending small step...")
                        # You can also do: bot.move_down(DESCENT_STEP_CM)
                        # But we keep RC loop; here we'll just nudge down
                        for _ in range(6):
                            rc_vx = 0; rc_vd = 0; rc_vyaw = 0; rc_vz = -18
                            time.sleep(0.08)
                        rc_vz = 0
                    except Exception as e:
                        print(f" Descent nudge failed: {e}")

                    # If close to ground, execute land
                    if height_cm is not None and height_cm < 40:
                        print(" Final land")
                        rc_vx = rc_vd = rc_vz = rc_vyaw = 0
                        try:
                            bot.land()
                        except Exception as e:
                            print(f" land() failed: {e}")
                        break
                else:
                    # keep aligning
                    pass
            else:
                # lost H during landing -> gentle hover and rotate to re-acquire
                rc_vx = rc_vd = rc_vz = 0
                rc_vyaw = 16

        # --------------- Gate assist  ---------------
       
        if gate_center is not None and STATE in ("TRACK_TAG","SEARCH_TAGS"):
            gx, gy = gate_center
            gex = float(gx - CX)
            gey = float(-(gy - CY))
            # Small correction (do not override strong PID fully)
            gvx = int(np.clip(0.25 * gex, -12, 12))
            gvz = int(np.clip(0.25 * gey, -12, 12))
            rc_vx = int(np.clip(rc_vx + gvx, -MAX_RC, MAX_RC))
            rc_vz = int(np.clip(rc_vz + gvz, -MAX_RC, MAX_RC))
            cv2.putText(vis, "GATE assist", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # --------------- UI / Diagnostics ---------------
        cv2.putText(vis, f"STATE: {STATE}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if battery is not None:
            cv2.putText(vis, f"Battery: {battery}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if height_cm is not None:
            cv2.putText(vis, f"H: {height_cm}cm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.circle(vis, (CX, CY), 6, (255,255,255), -1)

        # Show
        cv2.imshow("OUT", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(" Quit requested.")
            break

except KeyboardInterrupt:
    print(" Keyboard interrupt; landing...")

except Exception as e:
    print(f" Fatal error: {e}")


try:
    running = False
    time.sleep(0.1)
    bot.send_rc_control(0,0,0,0)
    bot.streamoff()
except: pass

try:
    bot.land()
except:
    pass

try:
    bot.end()
except:
    pass





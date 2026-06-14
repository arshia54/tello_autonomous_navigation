from djitellopy import tello
import cv2
import time
import logging
import at
import numpy as np
import PIDA1

# ---------------- ID OF TAGS ----------------
tag_ID = [0]
close_gate = [0,1,0,1]
# ---------------- DETECTION LAB/HSV VALUES ----------------
lab_values = {
    'l1_min': 30,  'l1_max': 130,
    'a1_min': 135, 'a1_max': 180,
    'b1_min': 105, 'b1_max': 165,

    'h_y_min': 20,  'h_y_max': 40,
    's_y_min': 80,  's_y_max': 255,
    'v_y_min': 120, 'v_y_max': 255,
}

# ---------------- DETECT GATE AND INNER OPENING ----------------
def detect_gate_and_opening(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_frame = np.array([
        lab_values['l1_min'],
        lab_values['a1_min'],     
        lab_values['b1_min']
    ])
    upper_frame = np.array([
        lab_values['l1_max'],
        lab_values['a1_max'],
        lab_values['b1_max']
    ])

    lower_yellow = np.array([
        lab_values['h_y_min'],
        lab_values['s_y_min'],
        lab_values['v_y_min']
    ])
    upper_yellow = np.array([
        lab_values['h_y_max'],
        lab_values['s_y_max'],
        lab_values['v_y_max']
    ])

    mask_frame = cv2.inRange(lab, lower_frame, upper_frame)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)

    mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_OPEN, kernel)
    mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_CLOSE, kernel)

    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    gate_mask = cv2.bitwise_or(mask_frame, mask_yellow)
    gate_mask = cv2.dilate(gate_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(gate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gate_found = False
    gate_cx, gate_cy = None, None
    gate_cnt = None
    gate_rect = None
    best_gate_area = 0

    h, w = frame.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 80 or ch < 80:
            continue

        if area > best_gate_area:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            gate_cx = int(M["m10"] / M["m00"])
            gate_cy = int(M["m01"] / M["m00"])
            gate_cnt = cnt
            gate_rect = (x, y, cw, ch)
            best_gate_area = area
            gate_found = True

    opening_found = False
    open_cx, open_cy = None, None
    open_cnt = None
    best_open_area = 0
    open_mask_roi_full = np.zeros((h, w), dtype=np.uint8)

    if gate_found:
        x, y, cw, ch = gate_rect

        roi_gate_mask = gate_mask[y:y+ch, x:x+cw]
        roi_open_mask = cv2.bitwise_not(roi_gate_mask)

        roi_open_mask = cv2.morphologyEx(roi_open_mask, cv2.MORPH_OPEN, kernel)
        roi_open_mask = cv2.morphologyEx(roi_open_mask, cv2.MORPH_CLOSE, kernel)

        roi_open_mask[0:5, :] = 0
        roi_open_mask[-5:, :] = 0
        roi_open_mask[:, 0:5] = 0
        roi_open_mask[:, -5:] = 0

        open_contours, _ = cv2.findContours(roi_open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in open_contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            ox, oy, ocw, och = cv2.boundingRect(cnt)
            if ocw < 20 or och < 20:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            local_cx = int(M["m10"] / M["m00"])
            local_cy = int(M["m01"] / M["m00"])

            if area > best_open_area:
                best_open_area = area
                opening_found = True
                open_cx = x + local_cx
                open_cy = y + local_cy

                cnt_global = cnt.copy()
                cnt_global[:, 0, 0] += x
                cnt_global[:, 0, 1] += y
                open_cnt = cnt_global

        open_mask_roi_full[y:y+ch, x:x+cw] = roi_open_mask

    debug_data = {
        "mask_frame": mask_frame,
        "mask_yellow": mask_yellow,
        "gate_mask": gate_mask,
        "open_mask": open_mask_roi_full,
        "gate_cnt": gate_cnt,
        "open_cnt": open_cnt,
        "gate_rect": gate_rect
    }

    return gate_found, gate_cx, gate_cy, opening_found, open_cx, open_cy, debug_data

# ---------------- SETUP ----------------
robot = tello.Tello()
robot.LOGGER.setLevel(logging.ERROR)
robot.connect()
robot.streamon()

btt = robot.get_battery()
if btt < 30:
    print("!!!!!!!!!!!!!!!!!!!!!!battery low!!!!!!!!!!!!!!!!!!!!!")
    robot.end()
    exit()

robot.takeoff()
time.sleep(3)

state = 'SEARCH'
tag_now = 0
time_to_see = 0

gom_mode = False
gom_time = 0.0
gom_masir = [0, 0, 0, 0]

min_Gate = 50
max_Gate = 155
ertefa_search = max_Gate
zaman = time.time()

hover_start_time = 0.0
detect_found_count = 0
gate_detect_count = 0
tune_ok_count = 0
stable_opening_count = 0
lost_opening_count = 0

pass_forward_dist = 130

# ---------------- PID PARAMETERS (DEFINED BUT NOT USED YET) ----------------
PID_DT = 0.02

pid_tag_x = PIDA1.PID(
    kp=0.45, ki=0.01, kd=0.08, dt=PID_DT,
    output_limits=(-25,25), integral_limits=(-60,60),
    deadband=3, setpoint_weight=1.0, derivative_filter_alpha=0.6
)

pid_tag_y = PIDA1.PID(
    kp=0.45, ki=0.01, kd=0.08, dt=PID_DT,
    output_limits=(-25,25), integral_limits=(-60,60),
    deadband=3, setpoint_weight=1.0, derivative_filter_alpha=0.6
)

pid_tag_z = PIDA1.PID(
    kp=0.35, ki=0.02, kd=0.05, dt=PID_DT,
    output_limits=(-30,30), integral_limits=(-80,80),
    deadband=2, setpoint_weight=1.0, derivative_filter_alpha=0.5
)

pid_tag_r = PIDA1.PID(
    kp=0.55, ki=0.02, kd=0.08, dt=PID_DT,
    output_limits=(-20,20), integral_limits=(-50,50),
    deadband=2, setpoint_weight=1.0, derivative_filter_alpha=0.6
)

pid_open_x = PIDA1.PID(
    kp=0.20, ki=0.015, kd=0.10, dt=PID_DT,
    output_limits=(-30,30), integral_limits=(-50,50),
    deadband=5, setpoint_weight=1.0, derivative_filter_alpha=0.7
)

pid_open_y = PIDA1.PID(
    kp=0.20, ki=0.015, kd=0.10, dt=PID_DT,
    output_limits=(-30,30), integral_limits=(-50,50),
    deadband=5, setpoint_weight=1.0, derivative_filter_alpha=0.7
)

# ---------------- TUNING PARAMS ----------------
CENTER_TOL = 28
TUNE_OK_FRAMES = 3
WAIT_STABLE_FRAMES = 6
LOST_OPENING_LIMIT = 10

while True:
    frame = cv2.resize(cv2.cvtColor(robot.get_frame_read().frame, cv2.COLOR_RGB2BGR), [960, 720])
    tags = at.get_tags(frame)
    copy_frame = frame.copy()
    battery = robot.get_battery()
    height = robot.get_distance_tof()

    # ---------------- STATE: TAG ----------------
    if state == 'TAG':
        seen = False

        for tag in tags:
            if tag[0] == tag_ID[tag_now]:
                error_x = tag[1]
                error_y = 0 - tag[2]
                error_z = tag[3] - 60
                error_r = tag[4]

                if abs(error_x) > 37:
                    speed_x = 21 if error_x > 0 else -21
                elif abs(error_x) > 27:
                    speed_x = 15 if error_x > 0 else -15
                elif error_x > 16:
                    speed_x = 11
                elif error_x < -16:
                    speed_x = -11
                else:
                    speed_x = 0

                if error_y > 20:
                    speed_y = 20
                elif error_y < -15:
                    speed_y = -20
                else:
                    speed_y = 0

                if abs(error_z) > 45:
                    speed_z = 25 if error_z > 0 else -25
                elif error_z > 20:
                    speed_z = 18
                elif error_z < -20:
                    speed_z = -18
                else:
                    speed_z = 0

                if error_r > 10:
                    speed_r = 15
                elif error_r < -10:
                    speed_r = -15
                else:
                    speed_r = 0

                cv2.circle(copy_frame, (tag[5], tag[6]), 25, (0, 0, 255), 3)
                robot.send_rc_control(int(speed_x), int(speed_z), int(speed_y), int(speed_r))

                time_to_see = time.time()
                seen = True
                gom_mode = True
                gom_masir = [error_x, error_z, error_y, error_r]
                break

        if not seen:
            if gom_mode:
                gom_mode = False
                gom_time = time.time()
                state = 'GOM'
            else:
                robot.send_rc_control(0, 0, 0, 0)
                if (time.time() - time_to_see) > 5.0:
                    state = 'SEARCH'
                    zaman = time.time()
        else:
            if speed_x == 0 and speed_y == 0 and speed_z == 0 and speed_r == 0:
                robot.send_rc_control(0, 0, 0, 0)
                state = 'HOVER_BEFORE_DETECT'
                hover_start_time = time.time()

    # ---------------- STATE: HOVER BEFORE DETECT ----------------
    elif state == 'HOVER_BEFORE_DETECT':
        robot.send_rc_control(0, 0, 0, 0)

        if time.time() - hover_start_time > 1.0:
            detect_found_count = 0
            gate_detect_count = 0
            tune_ok_count = 0
            stable_opening_count = 0
            lost_opening_count = 0
            state = 'DETECT_OPENING'

    # ---------------- STATE: DETECT OPENING ----------------
    elif state == 'DETECT_OPENING':
        robot.send_rc_control(0, 0, 0, 0)

        gate_found, gcx, gcy, opening_found, ocx, ocy, debug_data = detect_gate_and_opening(frame)

        cv2.imshow("mask_frame", debug_data["mask_frame"])
        cv2.imshow("mask_yellow", debug_data["mask_yellow"])
        cv2.imshow("gate_mask", debug_data["gate_mask"])
        cv2.imshow("open_mask", debug_data["open_mask"])

        if debug_data["gate_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["gate_cnt"]], -1, (0, 255, 255), 3)

        if debug_data["open_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["open_cnt"]], -1, (0, 255, 0), 3)

        if gate_found:
            gate_detect_count += 1
            cv2.circle(copy_frame, (gcx, gcy), 10, (0, 255, 255), -1)
        else:
            gate_detect_count = 0

        if opening_found:
            detect_found_count += 1
            cv2.circle(copy_frame, (ocx, ocy), 12, (255, 0, 0), -1)
        else:
            detect_found_count = 0

        if gate_detect_count >= 3 and detect_found_count >= 3:
            stable_opening_count = 0
            tune_ok_count = 0
            lost_opening_count = 0
            state = 'WAIT_STABLE_OPENING'

    # ---------------- STATE: WAIT STABLE OPENING ----------------
    elif state == 'WAIT_STABLE_OPENING':
        robot.send_rc_control(0, 0, 0, 0)

        gate_found, gcx, gcy, opening_found, ocx, ocy, debug_data = detect_gate_and_opening(frame)

        cv2.imshow("mask_frame", debug_data["mask_frame"])
        cv2.imshow("mask_yellow", debug_data["mask_yellow"])
        cv2.imshow("gate_mask", debug_data["gate_mask"])
        cv2.imshow("open_mask", debug_data["open_mask"])

        if debug_data["gate_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["gate_cnt"]], -1, (0, 255, 255), 3)

        if debug_data["open_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["open_cnt"]], -1, (0, 255, 0), 3)

        if opening_found:
            stable_opening_count += 1
            lost_opening_count = 0
            cv2.circle(copy_frame, (ocx, ocy), 12, (255, 0, 0), -1)
        else:
            stable_opening_count = 0
            lost_opening_count += 1

        cv2.putText(copy_frame, f"WAIT_STABLE_OPENING: {stable_opening_count}/{WAIT_STABLE_FRAMES}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        if stable_opening_count >= WAIT_STABLE_FRAMES:
            tune_ok_count = 0
            state = 'TUNE_OPENING_XY'

        if lost_opening_count >= LOST_OPENING_LIMIT:
            state = 'DETECT_OPENING'

    # ---------------- STATE: TUNE OPENING XY (با شرط‌های پله‌ای) ----------------
    elif state == 'TUNE_OPENING_XY':
        gate_found, gcx, gcy, opening_found, ocx, ocy, debug_data = detect_gate_and_opening(frame)

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        cv2.circle(copy_frame, (frame_center_x, frame_center_y), 8, (0, 0, 255), -1)

        cv2.imshow("mask_frame", debug_data["mask_frame"])
        cv2.imshow("mask_yellow", debug_data["mask_yellow"])
        cv2.imshow("gate_mask", debug_data["gate_mask"])
        cv2.imshow("open_mask", debug_data["open_mask"])

        if debug_data["gate_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["gate_cnt"]], -1, (0, 255, 255), 3)

        if debug_data["open_cnt"] is not None:
            cv2.drawContours(copy_frame, [debug_data["open_cnt"]], -1, (0, 255, 0), 3)

        if opening_found:
            lost_opening_count = 0

            cv2.circle(copy_frame, (ocx, ocy), 10, (255, 0, 0), -1)
            cv2.line(copy_frame, (frame_center_x, frame_center_y), (ocx, ocy), (255, 255, 0), 2)

            err_x = ocx - frame_center_x
            err_y = frame_center_y - ocy

            if abs(err_x) > 160:
                speed_x = 16 if err_x > 0 else -16
            elif abs(err_x) > 100:
                speed_x = 10 if err_x > 0 else -10
            elif abs(err_x) > 50:
                speed_x = 6 if err_x > 0 else -6
            elif abs(err_x) > CENTER_TOL:
                speed_x = 3 if err_x > 0 else -3
            else:
                speed_x = 0

            if abs(err_y) > 160:
                speed_y = 16 if err_y > 0 else -16
            elif abs(err_y) > 100:
                speed_y = 10 if err_y > 0 else -10
            elif abs(err_y) > 50:
                speed_y = 6 if err_y > 0 else -6
            elif abs(err_y) > CENTER_TOL:
                speed_y = 3 if err_y > 0 else -3
            else:
                speed_y = 0

            robot.send_rc_control(int(speed_x), 0, int(speed_y), 0)

            cv2.putText(copy_frame, f"err_x: {err_x}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(copy_frame, f"err_y: {err_y}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(copy_frame, f"tune_ok_count: {tune_ok_count}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

            print(f"TUNE XY -> err_x={err_x}, err_y={err_y}, sx={speed_x}, sy={speed_y}")

            if abs(err_x) <= CENTER_TOL and abs(err_y) <= CENTER_TOL:
                tune_ok_count += 1
                robot.send_rc_control(0, 0, 0, 0)
            else:
                tune_ok_count = 0

            if tune_ok_count >= TUNE_OK_FRAMES:
                robot.send_rc_control(0, 0, 0, 0)
                time.sleep(0.4)
                state = 'PASS_GATE'

        else:
            lost_opening_count += 1
            robot.send_rc_control(0, 0, 0, 0)

            if lost_opening_count >= LOST_OPENING_LIMIT:
                state = 'DETECT_OPENING'

    # ---------------- STATE: PASS GATE ----------------
    elif state == 'PASS_GATE':
        robot.send_rc_control(0, 0, 0, 0)
        time.sleep(0.4)

        try:
            robot.move_forward(pass_forward_dist)
        except:
            print("move_forward failed")

        tag_now += 1

        if tag_now >= len(tag_ID):
            robot.land()
            break

        zaman = time.time()

        if height > 80:
            ertefa_search = min_Gate
        else:
            ertefa_search = max_Gate

        state = 'SEARCH'

    # ---------------- STATE: GOM ----------------
    elif state == 'GOM':
        rx = ry = rz = rr = 0
        seen = False

        for tag in tags:
            if tag[0] == tag_ID[tag_now]:
                robot.send_rc_control(0, 0, 0, 0)
                seen = True
                state = 'TAG'
                break

        if time.time() - gom_time > 2.0:
            robot.send_rc_control(0, 0, 0, 0)
            zaman = time.time()
            state = 'SEARCH'
        else:
            if gom_masir[0] > 0:
                rx = 20
            elif gom_masir[0] < 0:
                rx = -20

            if gom_masir[2] < 0:
                ry = -15
            elif gom_masir[2] > 0:
                ry = 13

            rz = -25

            if gom_masir[3] < 0:
                rr = 10
            elif gom_masir[3] > 0:
                rr = -10

            robot.send_rc_control(int(rx), int(rz), int(ry), int(rr))

    # ---------------- STATE: SEARCH ----------------
    elif state == 'SEARCH':
        seen = False

        for tag in tags:
            if tag[0] == tag_ID[tag_now]:
                seen = True
                break

        if seen:
            robot.send_rc_control(0, 0, 0, 0)
            state = 'TAG'
        else:
            search_r = 40

            if time.time() - zaman > 18.0:
                if ertefa_search == max_Gate:
                    ertefa_search = min_Gate
                else:
                    ertefa_search = max_Gate
                zaman = time.time()
                speed_h = 0
            else:
                error_h = (height - ertefa_search) * (-1.0)
                speed_h = int(np.clip(error_h, -55, 55))
                if abs(error_h) < 10:
                    speed_h = 0

            robot.send_rc_control(0, 0, int(speed_h), int(search_r))

    # ---------------- UI ----------------
    cv2.putText(copy_frame, f"Battery: {battery}%", (20, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    cv2.putText(copy_frame, f"Mode: {state}", (520, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    cv2.putText(copy_frame, f"Height: {height}", (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

    cv2.imshow('OUT', copy_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

robot.land()
robot.end()
cv2.destroyAllWindows()

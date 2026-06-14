# Tello Drone Autonomous Gate Navigation with AprilTag & Computer Vision

An autonomous system for navigating a Tello drone through gates using AprilTag detection, color-based gate detection, and smooth Tanh-based velocity control.

---

## Features

- Gate detection using color filtering (LAB + HSV color spaces)
- Precise localization with AprilTag (distance and angle estimation)
- Smooth velocity control using Tanh function (no abrupt movements)
- GOM mode (estimated movement) when AprilTag is temporarily lost
- Support for multiple sequential gates (configurable tag_ID list)
- Low-pass filter on target coordinates for improved stability
- Real-time visualization (masks, contours, drone state)

---

## Technologies

| Library | Purpose |
|---------|---------|
| djitellopy | Tello drone communication |
| opencv-python | Image processing, color detection, contour finding |
| pupil-apriltags | AprilTag detection and pose estimation |
| scipy.spatial.transform | Rotation matrix to Euler angles conversion |
| numpy | Numerical operations and arrays |

---

## State Machine Logic

1. SEARCH - Rotate and search for AprilTag
2. APPROACH_TAG - Approach tag with distance and angle control
3. ALIGNED_AT_TAG - Brief pause
4. MOVE_BACK & MOVE_UP - Change viewing angle to see the gate
5. DETECT_GATE - Detect gate using color masks
6. WAIT_STABLE - Ensure target stability
7. TUNE_XY - Align drone to gate center using Tanh speed control
8. TUNE_VERIFY - Final position verification
9. PASS_GATE - Pass through gate and proceed to next tag

GOM mode = Estimated movement continuation when AprilTag is temporarily lost.

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- Tello drone connected to computer Wi-Fi
- AprilTag (family tag36h11) printed and mounted on the gate

### Install Dependencies

```bash
pip install djitellopy opencv-python pupil-apriltags scipy numpy

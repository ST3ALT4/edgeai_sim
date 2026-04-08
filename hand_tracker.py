# ─────────────────────────────────────────
#  hand_tracker.py  —  MediaPipe wrapper
#
#  JETSON NANO FIX:
#  Uses the classic mp.solutions.hands API (mediapipe <= 0.9.x)
#  instead of the new Tasks API (mediapipe >= 0.10.x) which has
#  no pre-built aarch64 wheel for Jetson Nano / JetPack 4.6.
#
#  Output dict format is identical to the original so main.py
#  and particle_system.py are unchanged.
# ─────────────────────────────────────────

import cv2
import mediapipe as mp
import math
from config import PINCH_THRESHOLD, WIDTH, HEIGHT


# ── Drawing helper ────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _draw_landmarks(frame, landmarks_px):
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 200, 100), 1)


# ── HandTracker ───────────────────────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        mp_hands = mp.solutions.hands
        # classic Hands solution — works on mediapipe 0.8.x / 0.9.x (aarch64 wheel)
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,            # 0 = lite model, faster on Nano
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        print("[INFO] MediaPipe Hands (Solutions API) initialised.")

    # ── Main process call ─────────────────────────────────────────────────
    def process(self, frame):
        """
        Returns { 'left': HandData | None, 'right': HandData | None }
        Same dict format as the original Tasks-API version.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark read-only for internal MediaPipe optimisation
        rgb_frame.flags.writeable = False
        result = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        left_data  = None
        right_data = None

        if result.multi_hand_landmarks is None:
            return {"left": None, "right": None}

        for lm_proto, handedness_proto in zip(
            result.multi_hand_landmarks,
            result.multi_handedness,
        ):
            # MediaPipe labels are MIRRORED for selfie view — flip them
            raw_label = handedness_proto.classification[0].label  # "Left" / "Right"
            label = "Right" if raw_label == "Left" else "Left"

            # Pixel coords for drawing
            landmarks_px = [
                (int(lm.x * w), int(lm.y * h)) for lm in lm_proto.landmark
            ]
            _draw_landmarks(frame, landmarks_px)

            data = self._extract(lm_proto.landmark, w, h, landmarks_px)

            if label == "Left":
                left_data = data
            else:
                right_data = data

        return {"left": left_data, "right": right_data}

    # ── Feature extraction ────────────────────────────────────────────────
    def _extract(self, lm_list, w, h, landmarks_px):
        pts = [(lm.x, lm.y) for lm in lm_list]   # normalised 0-1

        # Palm centre = average of wrist + MCP joints
        palm_x = sum(pts[i][0] for i in [0, 1, 5, 9, 13, 17]) / 6
        palm_y = sum(pts[i][1] for i in [0, 1, 5, 9, 13, 17]) / 6

        fingers_up   = self._count_fingers(pts)
        finger_count = sum(fingers_up)

        pinch_dist = math.dist(pts[4], pts[8])
        is_pinched = pinch_dist < PINCH_THRESHOLD

        # Wrist tilt angle
        wrist   = pts[0]
        mid_mcp = pts[9]
        angle_rad = math.atan2(
            mid_mcp[1] - wrist[1],
            mid_mcp[0] - wrist[0],
        )
        tilt_deg = math.degrees(angle_rad)

        fingertips_px = [landmarks_px[i] for i in [4, 8, 12, 16, 20]]

        return {
            "palm_norm":     (palm_x, palm_y),
            "palm_px":       (int(palm_x * w), int(palm_y * h)),
            "finger_count":  finger_count,
            "fingers_up":    fingers_up,
            "is_pinched":    is_pinched,
            "pinch_dist":    pinch_dist,
            "tilt_deg":      tilt_deg,
            "fingertips_px": fingertips_px,
            "index_tip_px":  fingertips_px[1],
        }

    def _count_fingers(self, pts):
        fingers = []
        # Thumb (compare X axis)
        fingers.append(pts[4][0] < pts[3][0])
        # Index → Pinky (tip above pip joint in Y = finger up)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(pts[tip][1] < pts[pip][1])
        return fingers

    def release(self):
        self._hands.close()

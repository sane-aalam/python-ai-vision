"""
Multi-Human / Multi-Hand Finger Counter using MediaPipe Tasks API.
Detects up to 6 hands simultaneously across multiple people in the camera frame.
Calculates individual finger counts per hand and grand total across all humans!
"""

import sys
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure model exists
from download_model import ensure_model_exists
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Maximum number of hands to track simultaneously (supports multiple people)
MAX_HANDS = 6

FINGER_TIPS = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]     # Reference joints
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Palette of distinct colors for different hands
HAND_COLORS = [
    (0, 255, 255),  # Cyan
    (0, 255, 128),  # Lime Green
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (255, 200, 0),  # Electric Blue
    (180, 100, 255),# Lavender
]


def setup_multi_hand_detector(max_hands: int = MAX_HANDS) -> vision.HandLandmarker:
    """Configures MediaPipe to detect multiple hands simultaneously."""
    model_path = ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=max_hands,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return vision.HandLandmarker.create_from_options(options)


def count_hand_fingers(
    pixel_landmarks: Dict[int, Tuple[int, int]]
) -> Tuple[int, List[bool], List[str]]:
    """
    Counts raised fingers for a single hand.
    Uses distance-based thumb metric (invariant to Left vs Right hand).
    """
    fingers_up = [False, False, False, False, False]

    # 1. Thumb Check: Distance from Thumb Tip (4) to Pinky Base (17)
    thumb_tip = np.array(pixel_landmarks[4], dtype=np.float32)
    thumb_ip = np.array(pixel_landmarks[3], dtype=np.float32)
    pinky_mcp = np.array(pixel_landmarks[17], dtype=np.float32)

    dist_tip_to_pinky = float(np.linalg.norm(thumb_tip - pinky_mcp))
    dist_ip_to_pinky = float(np.linalg.norm(thumb_ip - pinky_mcp))
    fingers_up[0] = dist_tip_to_pinky > dist_ip_to_pinky

    # 2. Four Fingers (Index, Middle, Ring, Pinky)
    for i in range(1, 5):
        tip_idx = FINGER_TIPS[i]
        pip_idx = FINGER_PIPS[i]
        if pixel_landmarks[tip_idx][1] < pixel_landmarks[pip_idx][1]:
            fingers_up[i] = True

    total = sum(fingers_up)
    names = [FINGER_NAMES[i] for i, is_up in enumerate(fingers_up) if is_up]
    return total, fingers_up, names


def get_hand_bounding_box(
    pixel_landmarks: Dict[int, Tuple[int, int]], pad: int = 15
) -> Tuple[int, int, int, int]:
    """Calculates enclosing bounding box (x_min, y_min, x_max, y_max) for a hand."""
    xs = [pt[0] for pt in pixel_landmarks.values()]
    ys = [pt[1] for pt in pixel_landmarks.values()]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def run_multi_hand_counter():
    """Main video loop tracking multiple humans and multiple hands in real-time."""
    print("=" * 65)
    print("     MULTI-HUMAN & MULTI-HAND FINGER COUNTER (UP TO 6 HANDS)     ")
    print("=" * 65)
    print(f"[System] Initializing Multi-Hand detector (Capacity: {MAX_HANDS} hands)...")

    detector = setup_multi_hand_detector(MAX_HANDS)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("[System] Multi-Hand Counter active! Step in front of the camera with multiple people.")
    print("[System] Press 'q' or 'ESC' to exit.\n")

    window_name = "Multi-Human Finger Counter [Press 'q' to Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Mirror horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe RGB conversion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            grand_total_fingers = 0
            num_hands_detected = 0

            # If hands are found
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                num_hands_detected = len(result.hand_landmarks)

                # Iterate through EVERY detected hand (from you, friends, or family!)
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # Pick color for this hand
                    hand_color = HAND_COLORS[hand_idx % len(HAND_COLORS)]

                    # Handedness label (Left vs Right) if available
                    label = f"Hand #{hand_idx + 1}"
                    if result.handedness and hand_idx < len(result.handedness):
                        handedness_category = result.handedness[hand_idx][0]
                        label += f" ({handedness_category.category_name})"

                    # Convert to pixel coordinates
                    pixel_map = {}
                    for idx, lm in enumerate(landmarks):
                        px = int(np.clip(lm.x * w, 0, w - 1))
                        py = int(np.clip(lm.y * h, 0, h - 1))
                        pixel_map[idx] = (px, py)

                    # Count fingers for this specific hand
                    hand_count, fingers_up, _ = count_hand_fingers(pixel_map)
                    grand_total_fingers += hand_count

                    # Draw hand bounding box
                    bx1, by1, bx2, by2 = get_hand_bounding_box(pixel_map)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), hand_color, 2)

                    # Floating badge directly above this hand
                    badge_text = f"{label}: {hand_count} Up"
                    badge_y = max(25, by1 - 10)
                    cv2.rectangle(
                        frame,
                        (bx1, badge_y - 20),
                        (bx1 + len(badge_text) * 10 + 10, badge_y + 4),
                        (20, 20, 25),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        badge_text,
                        (bx1 + 5, badge_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        hand_color,
                        1,
                        cv2.LINE_AA,
                    )

                    # Draw colored dots on tips (Green = extended, Red = folded)
                    for i, tip_idx in enumerate(FINGER_TIPS):
                        tip_pos = pixel_map[tip_idx]
                        tip_color = (0, 255, 0) if fingers_up[i] else (0, 0, 255)
                        cv2.circle(frame, tip_pos, 7, tip_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, tip_pos, 10, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Global Top HUD Banner ---
            banner_h = 56
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 24), -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
            cv2.line(frame, (0, banner_h), (w, banner_h), (0, 255, 255), 1)

            # Grand total fingers
            cv2.putText(
                frame,
                f"TOTAL FINGERS: {grand_total_fingers}",
                (16, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Hands count
            cv2.putText(
                frame,
                f"Hands: {num_hands_detected}/{MAX_HANDS}",
                (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )

            # Render display
            cv2.imshow(window_name, frame)

            # Exit check
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Multi-Hand Counter stopped cleanly.")


if __name__ == "__main__":
    run_multi_hand_counter()

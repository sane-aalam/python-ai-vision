"""
Finger Counter Module using MediaPipe Tasks API and OpenCV.

Step 1: Import modules and setup MediaPipe object
Step 2: Create finger counter function
Step 3: Create the main loop and run the project
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Ensure model exists
from download_model import ensure_model_exists
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================================================================
# STEP 1: IMPORT MODULES & SETUP MEDIAPIPE OBJECT
# ==============================================================================
def setup_mediapipe_hand_landmarker() -> vision.HandLandmarker:
    """Initializes and returns the MediaPipe HandLandmarker object."""
    model_path = ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


# ==============================================================================
# STEP 2: CREATING FINGER COUNTER FUNCTION
# ==============================================================================
# Landmark indices for tips and reference joints
FINGER_TIPS = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
FINGER_PIPS = [3, 6, 10, 14, 18]     # Thumb IP, Index PIP, Middle PIP, Ring PIP, Pinky PIP
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def count_raised_fingers(
    pixel_landmarks: Dict[int, Tuple[int, int]]
) -> Tuple[int, List[bool], List[str]]:
    """
    Determines how many and which fingers are raised/extended.

    Logic:
    - In image coordinates, Y = 0 is at the top.
    - If Finger Tip Y is SMALLER than PIP Y (Tip is higher up), the finger is EXTENDED.
    - For the Thumb, it moves horizontally: we check if Tip X is further outward than IP X
      relative to the Wrist (0) and Pinky MCP (17).

    Returns:
        total_count (int): Total number of extended fingers (0 to 5)
        fingers_up (List[bool]): True/False for [Thumb, Index, Middle, Ring, Pinky]
        raised_names (List[str]): Names of extended fingers
    """
    fingers_up = [False, False, False, False, False]

    # --- 1. Thumb Check (Universal: Works on BOTH Left & Right Hands) ---
    # We measure distance from Thumb Tip (4) to the base of the Pinky (17 - Pinky MCP).
    # When the thumb is extended outward, Tip 4 is farther from Pinky Base 17 than IP 3 is.
    # When folded into a fist, Tip 4 is tucked in close to Pinky Base 17.
    # This distance metric is 100% symmetric and works equally for Left and Right hands!
    thumb_tip = np.array(pixel_landmarks[4], dtype=np.float32)
    thumb_ip = np.array(pixel_landmarks[3], dtype=np.float32)
    pinky_mcp = np.array(pixel_landmarks[17], dtype=np.float32)

    dist_tip_to_pinky = float(np.linalg.norm(thumb_tip - pinky_mcp))
    dist_ip_to_pinky = float(np.linalg.norm(thumb_ip - pinky_mcp))

    fingers_up[0] = dist_tip_to_pinky > dist_ip_to_pinky

    # --- 2. Four Fingers Check (Index, Middle, Ring, Pinky) ---
    # A finger is up if its TIP Y is strictly above (smaller Y) its PIP Y
    for i in range(1, 5):
        tip_idx = FINGER_TIPS[i]
        pip_idx = FINGER_PIPS[i]
        if pixel_landmarks[tip_idx][1] < pixel_landmarks[pip_idx][1]:
            fingers_up[i] = True

    total_count = sum(fingers_up)
    raised_names = [FINGER_NAMES[i] for i, is_up in enumerate(fingers_up) if is_up]

    return total_count, fingers_up, raised_names


# ==============================================================================
# STEP 3: CREATING THE MAIN LOOP AND RUNNING THE PROJECT
# ==============================================================================
def run_finger_counter():
    """Main video loop capturing webcam feed and displaying live finger counts."""
    print("=" * 60)
    print("           REAL-TIME FINGER COUNTER (0 - 5)           ")
    print("=" * 60)
    print("[System] Initializing camera and MediaPipe...")

    detector = setup_mediapipe_hand_landmarker()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("[System] Webcam started! Press 'q' or 'ESC' to exit.")

    window_name = "AI Vision - Finger Counter [Press 'q' to Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Mirror horizontally for natural mirror-like viewing
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            count = 0
            raised_names = []

            # If hand is detected
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                landmarks = result.hand_landmarks[0]

                # Convert normalized coordinates [0.0, 1.0] to frame pixels
                pixel_map = {}
                for idx, lm in enumerate(landmarks):
                    px = int(np.clip(lm.x * w, 0, w - 1))
                    py = int(np.clip(lm.y * h, 0, h - 1))
                    pixel_map[idx] = (px, py)

                # Call our Step 2 Finger Counter Function!
                count, fingers_up, raised_names = count_raised_fingers(pixel_map)

                # Draw green dots on extended finger tips, red on folded ones
                for i, tip_idx in enumerate(FINGER_TIPS):
                    pos = pixel_map[tip_idx]
                    dot_color = (0, 255, 0) if fingers_up[i] else (0, 0, 255)
                    cv2.circle(frame, pos, 8, dot_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, pos, 12, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Visual HUD Display ---
            # 1. Semi-transparent black card in top-left corner
            card_w, card_h = 240, 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (20 + card_w, 20 + card_h), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (20, 20), (20 + card_w, 20 + card_h), (0, 255, 255), 2)

            # 2. Big bold counter number
            cv2.putText(
                frame,
                str(count),
                (35, 95),
                cv2.FONT_HERSHEY_DUPLEX,
                2.2,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )

            # 3. Label text
            cv2.putText(
                frame,
                "FINGERS UP",
                (100, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 4. Details (which fingers are up)
            fingers_text = ", ".join(raised_names) if raised_names else "Fist / None"
            cv2.putText(
                frame,
                fingers_text,
                (100, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

            # Show display window
            cv2.imshow(window_name, frame)

            # Exit check
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Finger Counter stopped cleanly.")


if __name__ == "__main__":
    run_finger_counter()

"""
Main application orchestrator for Virtual Mouse (Advanced Cyberpunk Edition).
Features:
- Real-time video processing loop
- Adaptive velocity cursor actuation
- Expanding shockwave click ripples
- Fading motion particle trails
- Dynamic pinch energy proximity meter
- Audio feedback ticks
"""

import sys
import cv2
from config import CAMERA, UI
from gesture_recognizer import GestureRecognizer, GestureType
from hand_detector import MediaPipeTasksDetector
from mouse_controller import MouseController
from ui_overlay import UIOverlay


def run_virtual_mouse() -> None:
    """Main application loop."""
    print("=" * 65)
    print("   VIRTUAL MOUSE: ADVANCED CYBERPUNK AI CONTROLLER   ")
    print("=" * 65)
    print("[System] Initializing camera feed and AI pipeline...")

    # Initialize webcam capture
    cap = cv2.VideoCapture(CAMERA.DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam at index {CAMERA.DEVICE_INDEX}.")
        print("Please check that your camera is connected and not in use by another app.")
        sys.exit(1)

    # Read actual capture dimensions
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAMERA.FRAME_WIDTH
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA.FRAME_HEIGHT
    print(f"[System] Camera successfully initialized ({actual_w}x{actual_h}).")

    # Initialize components
    detector = MediaPipeTasksDetector()
    mouse = MouseController(actual_w, actual_h)
    recognizer = GestureRecognizer()
    overlay = UIOverlay(actual_w, actual_h)

    print(f"[System] Screen resolution detected: {mouse.screen_w}x{mouse.screen_h}")
    print("\n[Controls Summary]")
    print("  • Move Cursor    : Point index finger (Landmark 8) inside active zone")
    print("  • Left Click     : Pinch Thumb (4) + Index (8) quickly (< 400ms)")
    print("  • Drag & Drop    : Pinch Thumb (4) + Index (8) and hold (>= 400ms)")
    print("  • Right Click    : Pinch Thumb (4) + Middle (12)")
    print("  • Scroll Up/Down : Two fingers (Index + Middle) extended moving vertically")
    print("  • Exit           : Press 'q' or 'ESC' key in the video window\n")

    window_name = "Virtual Mouse - Cyberpunk Edition [Press 'q' to Exit]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_drag_state = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Blank frame received from webcam. Continuing...")
                continue

            # Mirror frame horizontally for intuitive interaction
            if CAMERA.MIRROR_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # 1. Detect hand landmarks
            normalized_landmarks = detector.detect(frame)

            pixel_landmarks = None
            current_gesture = GestureType.IDLE
            hand_scale = 1.0

            if normalized_landmarks:
                # 2. Extract pixel coordinates and compute hand scale
                pixel_landmarks = detector.get_pixel_landmarks(
                    normalized_landmarks, actual_w, actual_h
                )
                hand_scale = detector.calculate_hand_scale(pixel_landmarks)

                # 3. Map index finger tip (8) to desktop screen coordinates with EMA and velocity acceleration
                index_tip_x, index_tip_y = pixel_landmarks[8]
                screen_x, screen_y = mouse.map_and_smooth(index_tip_x, index_tip_y)

                # 4. Classify gesture
                current_gesture, action_value = recognizer.process(
                    pixel_landmarks, hand_scale
                )

                # 5. Actuate mouse and trigger visual/audio ripples
                if current_gesture == GestureType.MOVE:
                    mouse.move(screen_x, screen_y)

                elif current_gesture == GestureType.LEFT_CLICK:
                    overlay.trigger_click_ripple(index_tip_x, index_tip_y, UI.COLOR_NEON_YELLOW)
                    mouse.lock_position((screen_x, screen_y))
                    mouse.left_click()
                    mouse.unlock_position()

                elif current_gesture == GestureType.DOUBLE_CLICK:
                    overlay.trigger_click_ripple(index_tip_x, index_tip_y, UI.COLOR_NEON_YELLOW)
                    mouse.lock_position((screen_x, screen_y))
                    mouse.double_click()
                    mouse.unlock_position()

                elif current_gesture == GestureType.RIGHT_CLICK:
                    overlay.trigger_click_ripple(index_tip_x, index_tip_y, UI.COLOR_MAGENTA)
                    mouse.lock_position((screen_x, screen_y))
                    mouse.right_click()
                    mouse.unlock_position()

                elif current_gesture == GestureType.DRAG:
                    if not prev_drag_state:
                        overlay.trigger_click_ripple(index_tip_x, index_tip_y, UI.COLOR_CRIMSON)
                    mouse.start_drag()
                    mouse.move(screen_x, screen_y)

                elif current_gesture == GestureType.DRAG_RELEASE:
                    overlay.trigger_click_ripple(index_tip_x, index_tip_y, UI.COLOR_CYAN)
                    mouse.stop_drag()

                elif current_gesture == GestureType.SCROLL:
                    if action_value != 0.0:
                        mouse.scroll(int(action_value))

                prev_drag_state = (current_gesture == GestureType.DRAG)

            else:
                # Hand exited camera frame
                if mouse.is_dragging:
                    mouse.stop_drag()
                prev_drag_state = False

            # 6. Render Cyberpunk HUD overlay
            display_frame = overlay.render(frame, pixel_landmarks, current_gesture, hand_scale)

            # 7. Show video window
            cv2.imshow(window_name, display_frame)

            # Check for user termination ('q' or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[System] Exit key pressed by user.")
                break

    except KeyboardInterrupt:
        print("\n[System] Interrupted by keyboard.")

    finally:
        # Graceful cleanup
        print("[System] Cleaning up resources...")
        mouse.stop_drag()
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Virtual Mouse stopped cleanly. Goodbye!")


if __name__ == "__main__":
    run_virtual_mouse()

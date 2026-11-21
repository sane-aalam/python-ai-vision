# Implementation Plan: 01 - Virtual Mouse (AI Hand-Tracking Mouse Controller)

> **Status:** Requirements Confirmed & Approved by User  
> **Target Scope:** Full Feature Suite (Movement, Left/Right Click, Drag & Drop, Scrolling)  
> **Topology:** MediaPipe 21 Hand Landmarks (Referencing [diagram.png](file:///d:/6%20Months%20-%20SDE2%20level/python-ai-vision/01-virtual-mouse/diagram.png))  
> **API Strategy:** Dual-compatible (MediaPipe Tasks API with `mp.solutions.hands` fallback)

---

## 1. Project Overview & Requirements Matrix
Build a production-grade, touchless mouse controller that translates webcam video input into real-time OS mouse actuation with jitter suppression and gesture recognition.

| Dimension | Specification | Implementation Detail |
| :--- | :--- | :--- |
| **Input Source** | Standard Webcam (Device 0) | Mirrored OpenCV video capture (`cv2.flip(frame, 1)`) at $\ge 30$ FPS. |
| **Detection Engine** | MediaPipe Tasks API + Fallback | 21 3D hand landmarks tracking; graceful fallback to classic `solutions.hands`. |
| **Cursor Anchor** | Landmark 8 (`INDEX_FINGER_TIP`) | Transformed through active ROI margin to screen coordinates. |
| **Coordinate Mapping** | NumPy linear interpolation | Maps camera margin boundary $[M, W - M]$ to $[0, W_{screen}]$. |
| **Motion Filter** | Exponential Moving Average (EMA) | Eliminates finger micro-jitters without introducing latency. |
| **OS Actuator** | PyAutoGUI | Zero-pause (`PAUSE = 0`), coordinate boundary clamping to prevent failsafe crashes. |

---

## 2. Confirmed Gesture Interaction Specification

Based on user selections, the following gestures are mapped:

```
                                  [Hand Landmarks]
                                         │
                 ┌───────────────────────┼────────────────────────┐
                 │                       │                        │
       [Landmark 8 Alone]     [Landmarks 4 & 8 Pinch]  [Landmarks 4 & 12 Pinch]
                 │                       │                        │
          Cursor Movement          Left Click / Drag          Right Click
                                         │
                                ┌────────┴────────┐
                                │                 │
                            < 400ms            >= 400ms
                         (Tap: Click)     (Hold: Drag & Drop)
```

1. **Cursor Navigation**:
   - Condition: Index finger (Landmark 8) extended, no active pinch.
   - Action: Cursor moves to filtered $(X_{screen}, Y_{screen})$.
2. **Left Click**:
   - Condition: Distance between Thumb Tip (Landmark 4) and Index Tip (Landmark 8) $< \text{threshold}$.
   - Action: `pyautogui.click()` executed once upon pinch contact (debounced). Position locked during click to avoid pointer drift.
3. **Double Click**:
   - Condition: Two consecutive thumb-index pinches within 350ms.
   - Action: `pyautogui.doubleClick()`.
4. **Drag & Drop**:
   - Condition: Thumb (4) and Index (8) pinch held continuously for $\ge 400\text{ms}$.
   - Action: `pyautogui.mouseDown()` on hold, continuous cursor tracking, `pyautogui.mouseUp()` upon release.
5. **Right Click**:
   - Condition: Distance between Thumb Tip (Landmark 4) and Middle Tip (Landmark 12) $< \text{threshold}$.
   - Action: `pyautogui.rightClick()`.
6. **Two-Finger Scroll**:
   - Condition: Both Index (8) and Middle (12) upright and separated from thumb.
   - Action: Vertical displacement $\Delta Y$ drives `pyautogui.scroll()`.

---

## 3. Architecture & Modular File Layout

```
01-virtual-mouse/
├── config.py             # System constants, screen sizes, thresholds, smoothing factors
├── hand_detector.py      # MediaPipe abstraction (Tasks API with Solutions fallback)
├── mouse_controller.py   # Coordinate scaling, EMA filter, PyAutoGUI actuation
├── gesture_recognizer.py # State machine tracking pinch states, hold durations, and debouncing
├── ui_overlay.py         # Visual feedback HUD (ROI bounding box, landmark dots, status tags, FPS)
├── main.py               # Main application loop orchestrating all modules
├── requirements.txt      # Dependency manifest
└── IMPLEMENTATION_PLAN.md# Technical specification & execution roadmap
```

---

## 4. Execution Phases

### Phase 1: Environment & Dependencies
- Verify and install packages (`opencv-python`, `mediapipe`, `pyautogui`, `numpy`).
- Verify webcam access and screen resolution via PyAutoGUI.

### Phase 2: Dual-Compatible Hand Detector (`hand_detector.py`)
- Encapsulate MediaPipe `HandLandmarker` Tasks API with auto-fallback to `mp.solutions.hands`.
- Provide landmark extraction in both normalized $[0.0, 1.0]$ and pixel dimensions.

### Phase 3: Screen Controller & Smoothing (`mouse_controller.py`)
- Screen dimension mapping with configurable boundary padding (active ROI).
- Exponential Moving Average (EMA) filter implementation to smooth pointer trajectory.

### Phase 4: Gesture State Machine (`gesture_recognizer.py`)
- Distance-based pinch detection normalized by hand size (Wrist to Index MCP).
- Debounce timers for click vs. drag detection.

### Phase 5: Visual HUD & Main Loop (`ui_overlay.py`, `main.py`)
- Overlay ROI rectangle, fingertip highlights, mode banners, and FPS counter.
- Clean shutdown handler on key press `'q'` or window close.

### Phase 6: End-to-End Verification
- Test corner-to-corner screen reachability.
- Verify click stability (no unwanted cursor drift during tap).
- Verify drag-and-drop and two-finger scroll functionality.

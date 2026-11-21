"""
Gesture recognition state machine.
Follows SOLID principles:
- Single Responsibility: Analyzes geometric landmark relationships and state timing.
- Open/Closed: New gestures can be added without altering detector or controller.
"""

import time
from enum import Enum, auto
from typing import Dict, Optional, Tuple
import numpy as np
from config import GESTURE


class GestureType(Enum):
    """Enumeration of recognized hand gestures."""
    IDLE = auto()
    MOVE = auto()
    LEFT_CLICK = auto()
    DOUBLE_CLICK = auto()
    DRAG = auto()
    DRAG_RELEASE = auto()
    RIGHT_CLICK = auto()
    SCROLL = auto()


class GestureRecognizer:
    """Classifies hand gestures based on normalized landmark distances and time state."""

    def __init__(self):
        # Timing trackers
        self._pinch_start_time: Optional[float] = None
        self._last_left_click_time: float = 0.0
        self._last_right_click_time: float = 0.0

        # State flags
        self._is_dragging: bool = False
        self._prev_scroll_y: Optional[int] = None
        self._pinch_active: bool = False

    def process(
        self, pixel_landmarks: Dict[int, Tuple[int, int]], hand_scale: float
    ) -> Tuple[GestureType, float]:
        """
        Processes current landmark frame and returns (GestureType, action_value).
        For SCROLL, action_value represents vertical scroll delta.
        For clicks/drags, action_value is 0.0.
        """
        now = time.time()

        # Landmark points
        thumb_tip = np.array(pixel_landmarks[4], dtype=np.float32)
        index_tip = np.array(pixel_landmarks[8], dtype=np.float32)
        middle_tip = np.array(pixel_landmarks[12], dtype=np.float32)

        index_pip = pixel_landmarks[6]
        middle_pip = pixel_landmarks[10]

        # Normalized distances relative to hand scale
        dist_thumb_index = float(np.linalg.norm(thumb_tip - index_tip)) / hand_scale
        dist_thumb_middle = float(np.linalg.norm(thumb_tip - middle_tip)) / hand_scale
        dist_index_middle = float(np.linalg.norm(index_tip - middle_tip)) / hand_scale

        # -------------------------------------------------------------
        # 1. Right Click: Thumb (4) pinched with Middle Finger (12)
        # -------------------------------------------------------------
        if dist_thumb_middle < GESTURE.RIGHT_CLICK_PINCH_RATIO:
            if (now - self._last_right_click_time) > GESTURE.CLICK_DEBOUNCE_TIME:
                self._last_right_click_time = now
                return GestureType.RIGHT_CLICK, 0.0

        # -------------------------------------------------------------
        # 2. Scroll Gesture: Index and Middle upright & adjacent
        # -------------------------------------------------------------
        is_index_extended = index_tip[1] < index_pip[1]
        is_middle_extended = middle_tip[1] < middle_pip[1]

        if (
            is_index_extended
            and is_middle_extended
            and dist_index_middle < GESTURE.SCROLL_TRIGGER_DIST_RATIO
            and dist_thumb_index > GESTURE.PINCH_CLICK_RATIO * 1.3
        ):
            current_y = int((index_tip[1] + middle_tip[1]) / 2)
            if self._prev_scroll_y is None:
                self._prev_scroll_y = current_y
                return GestureType.SCROLL, 0.0

            delta_y = self._prev_scroll_y - current_y  # Inverted: upward motion scrolls up
            self._prev_scroll_y = current_y

            if abs(delta_y) > GESTURE.SCROLL_DEADZONE_PIXELS:
                scroll_amount = delta_y * GESTURE.SCROLL_SPEED_MULTIPLIER
                return GestureType.SCROLL, float(scroll_amount)
            return GestureType.SCROLL, 0.0
        else:
            self._prev_scroll_y = None

        # -------------------------------------------------------------
        # 3. Left Click / Drag: Thumb (4) pinched with Index (8)
        # -------------------------------------------------------------
        if dist_thumb_index < GESTURE.PINCH_CLICK_RATIO:
            if not self._pinch_active:
                # Pinch just initiated
                self._pinch_active = True
                self._pinch_start_time = now

            hold_duration = now - (self._pinch_start_time or now)

            # If held longer than threshold, engage Drag mode
            if hold_duration >= GESTURE.DRAG_HOLD_DELAY:
                self._is_dragging = True
                return GestureType.DRAG, 0.0

            # Still in potential tap duration
            return GestureType.IDLE, 0.0

        else:
            # Pinch was released
            if self._pinch_active:
                self._pinch_active = False
                hold_duration = now - (self._pinch_start_time or now)
                self._pinch_start_time = None

                if self._is_dragging:
                    # Released out of drag mode
                    self._is_dragging = False
                    return GestureType.DRAG_RELEASE, 0.0

                # Short pinch release -> Click or Double Click
                if hold_duration < GESTURE.DRAG_HOLD_DELAY:
                    if (now - self._last_left_click_time) < GESTURE.DOUBLE_CLICK_MAX_DELAY:
                        self._last_left_click_time = 0.0  # Reset
                        return GestureType.DOUBLE_CLICK, 0.0
                    else:
                        self._last_left_click_time = now
                        return GestureType.LEFT_CLICK, 0.0

        # -------------------------------------------------------------
        # 4. Standard Cursor Navigation
        # -------------------------------------------------------------
        if self._is_dragging:
            return GestureType.DRAG, 0.0

        return GestureType.MOVE, 0.0

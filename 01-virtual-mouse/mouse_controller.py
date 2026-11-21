"""
Mouse controller module with Adaptive Velocity Acceleration and Audio Feedback.
Follows SOLID principles:
- Single Responsibility: Coordinate transformation, motion smoothing, velocity dynamics, and OS actuation.
- Interface Segregation: Clean minimal API for mouse actions.
"""

import math
import threading
import time
from typing import Tuple
import numpy as np
import pyautogui
from config import MOUSE

# Optional native Windows audio feedback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Configure PyAutoGUI for real-time responsiveness
pyautogui.PAUSE = MOUSE.PYAUTOGUI_PAUSE
pyautogui.FAILSAFE = True


def _play_click_audio_async() -> None:
    """Plays non-blocking audio tick on a background thread so the vision loop never drops frames."""
    if not HAS_WINSOUND or not MOUSE.ENABLE_AUDIO_FEEDBACK:
        return

    def _worker():
        try:
            winsound.Beep(MOUSE.AUDIO_CLICK_FREQ, MOUSE.AUDIO_CLICK_DUR)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


class MouseController:
    """Handles screen mapping, EMA smoothing, velocity acceleration, and OS actuation."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.screen_w, self.screen_h = pyautogui.size()

        # Previous position & timestamp for velocity computation
        self._prev_x: float = self.screen_w / 2.0
        self._prev_y: float = self.screen_h / 2.0
        self._prev_time: float = time.time()

        # State flags
        self.is_dragging: bool = False
        self._position_locked: bool = False
        self._locked_pos: Tuple[int, int] = (int(self._prev_x), int(self._prev_y))

    def map_and_smooth(self, finger_x: int, finger_y: int) -> Tuple[int, int]:
        """
        Maps fingertip coordinates within active ROI to screen coordinates.
        Applies Exponential Moving Average (EMA) and adaptive velocity acceleration.
        """
        if self._position_locked:
            return self._locked_pos

        now = time.time()
        dt = max(now - self._prev_time, 0.001)
        self._prev_time = now

        # Active ROI bounds
        roi_x_min = MOUSE.ROI_MARGIN_X
        roi_x_max = self.frame_w - MOUSE.ROI_MARGIN_X
        roi_y_min = MOUSE.ROI_MARGIN_Y
        roi_y_max = self.frame_h - MOUSE.ROI_MARGIN_Y

        # Linear mapping to screen dimensions
        target_x = np.interp(finger_x, (roi_x_min, roi_x_max), (0, self.screen_w))
        target_y = np.interp(finger_y, (roi_y_min, roi_y_max), (0, self.screen_h))

        # Calculate instantaneous velocity (pixels/second)
        delta_dist = math.hypot(target_x - self._prev_x, target_y - self._prev_y)
        velocity = delta_dist / dt

        # Adaptive velocity gain factor
        alpha = MOUSE.SMOOTHING_FACTOR
        if MOUSE.ENABLE_ACCELERATION:
            if velocity < MOUSE.VELOCITY_SLOW_THRESHOLD:
                # Precision Mode: dampen alpha for pixel-level micro-targeting
                alpha *= MOUSE.PRECISION_DAMPING
            elif velocity > MOUSE.VELOCITY_FAST_THRESHOLD:
                # Speed Boost Mode: increase alpha for fast monitor traversal
                alpha = min(1.0, alpha * MOUSE.BOOST_ACCELERATION_GAIN)

        # Exponential Moving Average (EMA)
        smooth_x = alpha * target_x + (1.0 - alpha) * self._prev_x
        smooth_y = alpha * target_y + (1.0 - alpha) * self._prev_y

        # Clamp inside screen boundaries
        pad = MOUSE.SCREEN_BORDER_PADDING
        clamped_x = int(np.clip(smooth_x, pad, self.screen_w - pad))
        clamped_y = int(np.clip(smooth_y, pad, self.screen_h - pad))

        self._prev_x = smooth_x
        self._prev_y = smooth_y
        return clamped_x, clamped_y

    def lock_position(self, current_pos: Tuple[int, int]) -> None:
        """Temporarily freezes cursor position during clicks to eliminate pointer drift."""
        self._position_locked = True
        self._locked_pos = current_pos

    def unlock_position(self) -> None:
        """Restores free cursor motion."""
        self._position_locked = False

    def move(self, x: int, y: int) -> None:
        """Moves cursor to screen coordinate."""
        try:
            pyautogui.moveTo(x, y)
        except pyautogui.FailSafeException:
            pass

    def left_click(self) -> None:
        """Executes a left mouse click with audio feedback."""
        _play_click_audio_async()
        try:
            pyautogui.click()
        except pyautogui.FailSafeException:
            pass

    def double_click(self) -> None:
        """Executes a double click with audio feedback."""
        _play_click_audio_async()
        try:
            pyautogui.doubleClick()
        except pyautogui.FailSafeException:
            pass

    def right_click(self) -> None:
        """Executes a right click with audio feedback."""
        _play_click_audio_async()
        try:
            pyautogui.rightClick()
        except pyautogui.FailSafeException:
            pass

    def start_drag(self) -> None:
        """Engages mouse drag."""
        if not self.is_dragging:
            _play_click_audio_async()
            try:
                pyautogui.mouseDown()
                self.is_dragging = True
            except pyautogui.FailSafeException:
                pass

    def stop_drag(self) -> None:
        """Releases mouse drag."""
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
                self.is_dragging = False
            except pyautogui.FailSafeException:
                pass

    def scroll(self, scroll_amount: int) -> None:
        """Executes vertical scrolling."""
        try:
            pyautogui.scroll(scroll_amount)
        except pyautogui.FailSafeException:
            pass

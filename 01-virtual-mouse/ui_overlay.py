"""
Sci-Fi Cyberpunk HUD overlay module.
Features:
- Semi-transparent glassmorphic HUD header
- Expanding click ripple shockwaves
- Neon fading motion particle ribbon
- Dynamic pinch proximity energy meter
- Anti-aliased holographic active control brackets
"""

import collections
import time
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
from config import MOUSE, UI
from gesture_recognizer import GestureType

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class UIOverlay:
    """Cyberpunk holographic HUD renderer with particle trails and shockwaves."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Motion trail buffer
        self._trail: Deque[Tuple[int, int]] = collections.deque(maxlen=UI.TRAIL_MAX_POINTS)

        # Active ripple effects: list of dicts {x, y, radius, color}
        self._ripples: List[Dict] = []

        # FPS calculation
        self._prev_frame_time = time.time()
        self._fps: float = 0.0

    def trigger_click_ripple(self, x: int, y: int, color: Tuple[int, int, int] = UI.COLOR_NEON_YELLOW) -> None:
        """Spawns an expanding shockwave ripple at given (x, y) coordinates."""
        self._ripples.append({
            "x": x,
            "y": y,
            "radius": 6,
            "color": color,
            "alpha": 1.0
        })

    def calculate_fps(self) -> float:
        """Calculates smoothed rolling FPS."""
        now = time.time()
        dt = now - self._prev_frame_time
        self._prev_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        return self._fps

    def draw_roi_box(self, frame: np.ndarray) -> None:
        """Renders futuristic holographic brackets defining the active interaction zone."""
        x1 = MOUSE.ROI_MARGIN_X
        y1 = MOUSE.ROI_MARGIN_Y
        x2 = self.frame_w - MOUSE.ROI_MARGIN_X
        y2 = self.frame_h - MOUSE.ROI_MARGIN_Y

        # Subtle boundary guide
        cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 60), 1)

        # Glowing holographic corners
        corner_len = 24
        thick = 2
        color = UI.COLOR_CYAN

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick, cv2.LINE_AA)

        # Reticle label
        cv2.putText(
            frame,
            "[ ACTIVE DESKTOP MATRIX ]",
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_motion_trail(self, frame: np.ndarray) -> None:
        """Renders a fading neon particle ribbon tracing index fingertip trajectory."""
        if len(self._trail) < 2:
            return

        trail_len = len(self._trail)
        for i in range(1, trail_len):
            # Calculate gradient thickness and alpha intensity
            factor = i / trail_len
            thickness = max(1, int(factor * 4))
            intensity = int(factor * 255)
            # Electric cyan-to-green trail
            color = (intensity, 255, int((1.0 - factor) * 200))
            cv2.line(frame, self._trail[i - 1], self._trail[i], color, thickness, cv2.LINE_AA)

    def update_and_draw_ripples(self, frame: np.ndarray) -> None:
        """Updates and renders expanding circular shockwave ripples."""
        surviving_ripples = []
        for r in self._ripples:
            r["radius"] += UI.RIPPLE_EXPANSION_SPEED
            if r["radius"] < UI.RIPPLE_MAX_RADIUS:
                fade = 1.0 - (r["radius"] / UI.RIPPLE_MAX_RADIUS)
                thick = max(1, int(fade * 3))
                # Outer pulse ring
                cv2.circle(frame, (r["x"], r["y"]), r["radius"], r["color"], thick, cv2.LINE_AA)
                # Inner echo ring
                if r["radius"] > 10:
                    cv2.circle(frame, (r["x"], r["y"]), int(r["radius"] * 0.55), r["color"], 1, cv2.LINE_AA)
                surviving_ripples.append(r)
        self._ripples = surviving_ripples

    def draw_pinch_energy_gauge(
        self,
        frame: np.ndarray,
        thumb_pos: Tuple[int, int],
        index_pos: Tuple[int, int],
        hand_scale: float,
        gesture: GestureType,
    ) -> None:
        """Draws a live dynamic energy bar between thumb and index showing click proximity."""
        tx, ty = thumb_pos
        ix, iy = index_pos
        mx, my = (tx + ix) // 2, (ty + iy) // 2

        dist = float(np.linalg.norm(np.array(thumb_pos) - np.array(index_pos)))
        ratio = dist / max(hand_scale, 1.0)

        # Proximity percentage: 0.0 (far) to 1.0 (touching)
        proximity = float(np.clip(1.0 - (ratio / 0.35), 0.0, 1.0))

        # Color transitions dynamically: Cyan (0%) -> Yellow (60%) -> Neon Red (100%)
        if proximity > 0.85:
            bar_color = UI.COLOR_CRIMSON
        elif proximity > 0.45:
            bar_color = UI.COLOR_NEON_YELLOW
        else:
            bar_color = UI.COLOR_CYAN

        # Connecting laser line
        cv2.line(frame, (tx, ty), (ix, iy), bar_color, 1, cv2.LINE_AA)

        # Midway energy indicator dot
        dot_radius = int(3 + proximity * 6)
        cv2.circle(frame, (mx, my), dot_radius, bar_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), dot_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_skeleton(
        self, frame: np.ndarray, pixel_landmarks: Dict[int, Tuple[int, int]]
    ) -> None:
        """Draws holographic skeleton links."""
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
                cv2.line(
                    frame,
                    pixel_landmarks[start_idx],
                    pixel_landmarks[end_idx],
                    (90, 90, 110),
                    1,
                    cv2.LINE_AA,
                )

        # Minor joint nodes
        for idx in range(21):
            if idx not in (4, 8, 12) and idx in pixel_landmarks:
                cv2.circle(frame, pixel_landmarks[idx], 2, (150, 150, 170), -1)

    def draw_fingertips(
        self, frame: np.ndarray, pixel_landmarks: Dict[int, Tuple[int, int]]
    ) -> None:
        """Draws highlighted glowing fingertip reticles."""
        index = pixel_landmarks.get(8)
        thumb = pixel_landmarks.get(4)
        middle = pixel_landmarks.get(12)

        if index:
            self._trail.append(index)
            # Glowing double-ring reticle for pointer tip
            cv2.circle(frame, index, 8, UI.COLOR_NEON_GREEN, -1, cv2.LINE_AA)
            cv2.circle(frame, index, 12, UI.COLOR_CYAN, 1, cv2.LINE_AA)
            cv2.circle(frame, index, 16, (255, 255, 255), 1, cv2.LINE_AA)

        if thumb:
            cv2.circle(frame, thumb, 6, UI.COLOR_NEON_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(frame, thumb, 10, (255, 255, 255), 1, cv2.LINE_AA)

        if middle:
            cv2.circle(frame, middle, 6, UI.COLOR_MAGENTA, -1, cv2.LINE_AA)
            cv2.circle(frame, middle, 9, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_glass_header(
        self, frame: np.ndarray, gesture: GestureType, fps: float
    ) -> None:
        """Renders semi-transparent frosted glass header with glowing status badge."""
        header_h = 46
        # Alpha blend frosted dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_w, header_h), UI.COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Neon divider line
        cv2.line(frame, (0, header_h), (self.frame_w, header_h), UI.COLOR_CYAN, 1, cv2.LINE_AA)

        # FPS readout
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            UI.COLOR_NEON_GREEN,
            2,
            cv2.LINE_AA,
        )

        # Mode status badge
        mode_text, badge_color = self._get_mode_badge(gesture)
        cv2.putText(
            frame,
            f"MODE: {mode_text}",
            (115, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        # Exit shortcut
        cv2.putText(
            frame,
            "Press 'q' to Quit",
            (self.frame_w - 135, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _get_mode_badge(self, gesture: GestureType) -> Tuple[str, Tuple[int, int, int]]:
        """Maps gesture type to descriptive label and glowing color."""
        mapping = {
            GestureType.IDLE: ("READY (NO HAND)", (130, 130, 140)),
            GestureType.MOVE: ("TRACKING", UI.COLOR_NEON_GREEN),
            GestureType.LEFT_CLICK: ("⚡ LEFT CLICK", UI.COLOR_NEON_YELLOW),
            GestureType.DOUBLE_CLICK: ("⚡⚡ DOUBLE CLICK", UI.COLOR_NEON_YELLOW),
            GestureType.DRAG: ("🔒 DRAGGING", UI.COLOR_CRIMSON),
            GestureType.DRAG_RELEASE: ("🔓 DRAG RELEASE", UI.COLOR_CYAN),
            GestureType.RIGHT_CLICK: ("⚡ RIGHT CLICK", UI.COLOR_MAGENTA),
            GestureType.SCROLL: ("↕ SCROLLING", UI.COLOR_CYAN),
        }
        return mapping.get(gesture, ("ACTIVE", UI.COLOR_WHITE))

    def render(
        self,
        frame: np.ndarray,
        pixel_landmarks: Optional[Dict[int, Tuple[int, int]]],
        gesture: GestureType,
        hand_scale: float = 1.0,
    ) -> np.ndarray:
        """Renders complete HUD pipeline."""
        fps = self.calculate_fps()

        # 1. Active desktop matrix boundary
        self.draw_roi_box(frame)

        # 2. Hand tracking, particle trail, and energy meter
        if pixel_landmarks:
            self.draw_skeleton(frame, pixel_landmarks)
            self.draw_motion_trail(frame)
            self.draw_fingertips(frame, pixel_landmarks)

            thumb = pixel_landmarks.get(4)
            index = pixel_landmarks.get(8)
            if thumb and index:
                self.draw_pinch_energy_gauge(frame, thumb, index, hand_scale, gesture)
        else:
            # Clear trail when hand leaves camera
            self._trail.clear()

        # 3. Expanding shockwave pulses
        self.update_and_draw_ripples(frame)

        # 4. Translucent glass header
        self.draw_glass_header(frame, gesture, fps)

        return frame

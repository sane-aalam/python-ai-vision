"""
Hand detection module using MediaPipe Tasks API.
Follows SOLID principles:
- Single Responsibility: Only extracts 21 hand landmarks from frames.
- Liskov Substitution: Extends BaseHandDetector abstract interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import MODEL
from download_model import ensure_model_exists


class BaseHandDetector(ABC):
    """Abstract interface for hand landmark detectors (LSP & DIP compliance)."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """Detects hand in a frame and returns 21 normalized landmarks (x, y, z)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Frees underlying detector resources."""
        pass


class MediaPipeTasksDetector(BaseHandDetector):
    """Concrete HandLandmarker implementation using modern MediaPipe Tasks API."""

    def __init__(self):
        # Ensure model bundle exists
        model_path = ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MODEL.NUM_HANDS,
            min_hand_detection_confidence=MODEL.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MODEL.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MODEL.MIN_TRACKING_CONFIDENCE,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Processes a BGR image frame and returns normalized (x, y, z) for 21 hand landmarks.
        Returns None if no hand is detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        # Extract the primary hand landmarks (list of 21 landmark objects)
        primary_hand = result.hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in primary_hand]

    def release(self) -> None:
        """Closes detector resources."""
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()

    @staticmethod
    def get_pixel_landmarks(
        landmarks: List[Tuple[float, float, float]], frame_w: int, frame_h: int
    ) -> Dict[int, Tuple[int, int]]:
        """Converts normalized landmarks [0.0, 1.0] into frame pixel coordinates (x, y)."""
        pixel_map: Dict[int, Tuple[int, int]] = {}
        for idx, (norm_x, norm_y, _) in enumerate(landmarks):
            px = int(np.clip(norm_x * frame_w, 0, frame_w - 1))
            py = int(np.clip(norm_y * frame_h, 0, frame_h - 1))
            pixel_map[idx] = (px, py)
        return pixel_map

    @staticmethod
    def calculate_hand_scale(pixel_landmarks: Dict[int, Tuple[int, int]]) -> float:
        """
        Calculates baseline hand scale: Euclidean distance between Wrist (0) and Index MCP (5).
        Used to make gesture thresholds distance-invariant (closer vs further from camera).
        """
        wrist = np.array(pixel_landmarks[0], dtype=np.float32)
        index_mcp = np.array(pixel_landmarks[5], dtype=np.float32)
        dist = float(np.linalg.norm(wrist - index_mcp))
        return max(dist, 1.0)  # Avoid zero-division

    @staticmethod
    def count_raised_fingers(pixel_landmarks: Dict[int, Tuple[int, int]]) -> Tuple[int, List[bool]]:
        """
        Counts extended fingers (0-5) by comparing tip Y positions to PIP Y positions.
        Returns (count, [thumb_up, index_up, middle_up, ring_up, pinky_up]).
        """
        fingers_up = [False, False, False, False, False]
        # Thumb: Distance from Thumb Tip (4) to Pinky MCP (17) vs Thumb IP (3) to Pinky MCP (17).
        # This distance metric is 100% invariant to Left vs. Right hand orientation.
        thumb_tip = np.array(pixel_landmarks[4], dtype=np.float32)
        thumb_ip = np.array(pixel_landmarks[3], dtype=np.float32)
        pinky_mcp = np.array(pixel_landmarks[17], dtype=np.float32)
        fingers_up[0] = bool(np.linalg.norm(thumb_tip - pinky_mcp) > np.linalg.norm(thumb_ip - pinky_mcp))

        # 4 Fingers: Tip Y < PIP Y
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
            if pixel_landmarks[tip][1] < pixel_landmarks[pip][1]:
                fingers_up[i + 1] = True

        return sum(fingers_up), fingers_up


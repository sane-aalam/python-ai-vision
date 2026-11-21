"""
Configuration module for Virtual Mouse application.
Contains all system constants, visual HUD themes, velocity acceleration parameters, and audio settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    """Camera capture and display parameters."""
    DEVICE_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 60
    MIRROR_HORIZONTAL: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """MediaPipe HandLandmarker model parameters."""
    MODEL_DIR: Path = Path(__file__).parent / "models"
    MODEL_FILENAME: str = "hand_landmarker.task"
    MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    NUM_HANDS: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.6
    MIN_PRESENCE_CONFIDENCE: float = 0.6
    MIN_TRACKING_CONFIDENCE: float = 0.6

    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_FILENAME


@dataclass(frozen=True)
class MouseConfig:
    """Screen coordinate mapping, EMA smoothing, and dynamic velocity acceleration."""
    # Active ROI margin (padding inside camera frame mapped to full desktop)
    ROI_MARGIN_X: int = 80
    ROI_MARGIN_Y: int = 60

    # Exponential Moving Average base smoothing factor
    SMOOTHING_FACTOR: float = 0.45

    # Adaptive velocity acceleration settings
    ENABLE_ACCELERATION: bool = True
    VELOCITY_SLOW_THRESHOLD: float = 300.0   # Pixels/sec for micro-precision damping
    VELOCITY_FAST_THRESHOLD: float = 1200.0  # Pixels/sec for speed-boost acceleration
    PRECISION_DAMPING: float = 0.65          # Slow motion sensitivity multiplier
    BOOST_ACCELERATION_GAIN: float = 1.35     # Fast flick sensitivity multiplier

    # Safe screen border margin to prevent triggering PyAutoGUI failsafe
    SCREEN_BORDER_PADDING: int = 2

    # Zero pause for 60 FPS actuation
    PYAUTOGUI_PAUSE: float = 0.0

    # Audio feedback
    ENABLE_AUDIO_FEEDBACK: bool = True
    AUDIO_CLICK_FREQ: int = 2400  # Hz
    AUDIO_CLICK_DUR: int = 18     # ms


@dataclass(frozen=True)
class GestureConfig:
    """Thresholds and timing for gesture recognition."""
    PINCH_CLICK_RATIO: float = 0.28
    RIGHT_CLICK_PINCH_RATIO: float = 0.28

    CLICK_DEBOUNCE_TIME: float = 0.25
    DOUBLE_CLICK_MAX_DELAY: float = 0.35
    DRAG_HOLD_DELAY: float = 0.40

    SCROLL_TRIGGER_DIST_RATIO: float = 0.30
    SCROLL_SPEED_MULTIPLIER: int = 25
    SCROLL_DEADZONE_PIXELS: float = 4.0


@dataclass(frozen=True)
class UIConfig:
    """Cyberpunk Sci-Fi Neon styling constants (BGR format)."""
    # Neon palette
    COLOR_CYAN = (255, 255, 0)
    COLOR_ELECTRIC_BLUE = (255, 200, 0)
    COLOR_NEON_GREEN = (0, 255, 128)
    COLOR_NEON_YELLOW = (0, 230, 255)
    COLOR_CRIMSON = (40, 40, 255)
    COLOR_MAGENTA = (255, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_HUD_BG = (18, 18, 24)

    # Particle trail & ripples
    TRAIL_MAX_POINTS: int = 14
    RIPPLE_EXPANSION_SPEED: int = 3
    RIPPLE_MAX_RADIUS: int = 40

    FONT_SCALE: float = 0.65
    FONT_THICKNESS: int = 2


# Global configuration instances
CAMERA = CameraConfig()
MODEL = ModelConfig()
MOUSE = MouseConfig()
GESTURE = GestureConfig()
UI = UIConfig()

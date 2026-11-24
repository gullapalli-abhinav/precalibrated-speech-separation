"""Editable starting values for the offline prototype."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEMALE = ROOT / "female_voice_sample.waw.wav"
MALE = ROOT / "male_voice_sample.wav"
REAL = ROOT / "test.wav"
LEGACY_DIGITAL = ROOT / "Digi_added_samples.wav"
OUTPUT = ROOT / "outputs"
SPEAKERS = ("female", "male")
METHODS = ("harmonic", "harmonic_lpc")
DEFAULT_METHOD = "harmonic_lpc"
MODES = ("mute-female", "mute-male", "both", "mute-both")


@dataclass(frozen=True)
class Settings:
    rate: int = 16000
    window: int = 640
    hop: int = 160
    fft: int = 1024
    pitch_min: float = 70
    pitch_max: float = 350
    pitch_step: float = 2.5
    lpc_order: int = 14
    calibration_fraction: float = 0.60
    guard_seconds: float = 0.25
    min_split_seconds: float = 2.0
    quiet_db: float = -42.0
    unvoiced_zcr: float = 0.18
    fallback_female: float = 0.50

    def __post_init__(self) -> None:
        if (
            self.rate != 16000 or not 0 < self.hop <= self.window // 2
            or self.fft < self.window or self.window % 2
            or not self.rate / (self.window // 2) < self.pitch_min < self.pitch_max < self.rate / 2
            or not 0 < self.pitch_step <= self.pitch_max - self.pitch_min
            or not 0 < self.lpc_order < self.window or not 0 < self.calibration_fraction < 1
            or self.guard_seconds < 0 or self.min_split_seconds <= 0
            or not 0 < self.unvoiced_zcr < 1 or not 0 <= self.fallback_female <= 1
        ):
            raise ValueError("Invalid DSP settings")


DEFAULT = Settings()

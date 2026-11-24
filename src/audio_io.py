"""WAV input, recording inspection and reproducible source partitions."""

from hashlib import sha256
from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import Settings

Audio = NDArray[np.float64]


def rms(x: Audio) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def resample(x: Audio, source_rate: int, rate: int) -> Audio:
    if source_rate == rate:
        return x
    divisor = gcd(source_rate, rate)
    return resample_poly(x, rate // divisor, source_rate // divisor)


def read_audio(path: Path, rate: int | None = 16000) -> tuple[Audio, dict]:
    try:
        source_rate, raw = wavfile.read(path)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{path}: expected a readable PCM or IEEE float WAV ({exc})") from exc
    if not len(raw) or raw.ndim not in (1, 2):
        raise ValueError(f"{path}: empty or unsupported WAV")
    x = raw.astype(np.float64)
    if raw.dtype.kind in "ui":
        limits = np.iinfo(raw.dtype)
        rails = np.count_nonzero((raw == limits.min) | (raw == limits.max))
        scale = 2.0 ** (limits.bits - 1)
        x = (x - (scale if raw.dtype.kind == "u" else 0)) / scale
    elif raw.dtype.kind == "f":
        rails = np.count_nonzero(np.abs(x) >= 1)
    else:
        raise ValueError(f"{path}: unsupported WAV sample type {raw.dtype}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: WAV contains non-finite samples")
    info = {
        "file": str(path.resolve()),
        "rate_hz": source_rate,
        "channels": raw.shape[1] if raw.ndim == 2 else 1,
        "decoded_dtype": str(raw.dtype),
        "duration_seconds": len(raw) / source_rate,
        "peak": float(np.max(np.abs(x))),
        "rms": rms(x),
        "rail_samples": int(rails),
    }
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate is not None:
        x = resample(x, source_rate, rate)
    return x, info


def write_audio(path: Path, x: Audio, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Float WAV preserves the separator's gain and avoids integer saturation.
    wavfile.write(path, rate, x.astype(np.float32))


def partition(path: Path, cfg: Settings) -> tuple[Audio, Audio, dict]:
    x, info = read_audio(path, rate=None)
    source_rate = info["rate_hz"]
    end_cal = int(len(x) * cfg.calibration_fraction)
    start_test = end_cal + round(cfg.guard_seconds * source_rate)
    if min(end_cal, len(x) - start_test) < cfg.min_split_seconds * source_rate:
        raise ValueError(
            f"{path.name}: {len(x) / source_rate:.2f} s is too short for disjoint "
            f"{cfg.min_split_seconds:g} s segments and the {cfg.guard_seconds:g} s guard"
        )
    # Reset the resampling filter at each boundary so its taps never span splits.
    calibration = resample(x[:end_cal], source_rate, cfg.rate)
    held_out = resample(x[start_test:], source_rate, cfg.rate)
    start_at_processing_rate = (start_test * cfg.rate + source_rate - 1) // source_rate
    info.update({
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "processing_rate_hz": cfg.rate,
        "source_calibration_samples": [0, end_cal],
        "source_held_out_samples": [start_test, len(x)],
        "calibration_samples": [0, len(calibration)],
        "held_out_samples": [start_at_processing_rate, start_at_processing_rate + len(held_out)],
    })
    return calibration, held_out, info

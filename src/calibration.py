"""Create, validate and load fixed speaker profiles."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from config import SPEAKERS, Settings
from src.audio_io import partition
from src.cepstrum import log_envelope
from src.lpc_analysis import envelope_bank
from src.pitch_tracking import frames, single_pitch


def write_json(path: Path, record: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(values: np.ndarray) -> dict:
    return dict(zip(("p05", "p25", "median", "p75", "p95"),
                    np.percentile(values, [5, 25, 50, 75, 95]).tolist()))


def band_starts(cfg: Settings) -> np.ndarray:
    """Boundaries in FFT bins for matching short-time spectral envelopes."""
    bins = np.geomspace(75, cfg.rate / 2, 31) * cfg.fft / cfg.rate
    return np.unique(np.r_[0, np.rint(bins[:-1]).astype(int)])


def spectral_examples(data, voiced: np.ndarray, cfg: Settings, *, cepstral: bool = False) -> list:
    indices = []
    for group in (data.active & voiced, data.active & ~voiced):
        hits = np.flatnonzero(group)
        if hits.size:
            indices.extend(hits[np.linspace(0, hits.size - 1, min(16, hits.size), dtype=int)])
    spectra = data.spectrum[indices]
    power = np.exp(2*log_envelope(spectra, cfg, 2.)) if cepstral else np.abs(spectra) ** 2
    bands = np.add.reduceat(power, band_starts(cfg), axis=1)
    return (bands / np.maximum(bands.sum(axis=1, keepdims=True), 1e-12)).tolist()


def calibrate_speaker(x: np.ndarray, info: dict, cfg: Settings, *, include_cepstral: bool = False) -> tuple[dict, tuple]:
    data = frames(x, cfg)
    f0, cepstral, confidence = single_pitch(data, cfg)
    voiced = f0 > 0
    if np.count_nonzero(voiced) < 12:
        raise ValueError(f"{info['file']}: fewer than 12 confidently voiced calibration frames")
    pitch = f0[voiced]
    log_pitch = np.log2(pitch)
    agree = voiced & (cepstral > 0)
    median = float(np.median(pitch))
    strength = np.abs(data.spectrum[voiced])
    frequencies = np.arange(strength.shape[1]) * cfg.rate / cfg.fft
    harmonic = np.arange(1, 13)
    bins = np.rint(pitch[:, None] * harmonic / (cfg.rate / cfg.fft)).astype(int)
    bins = np.minimum(bins, len(frequencies) - 1)
    relative = np.take_along_axis(strength, bins, axis=1) / np.maximum(strength.max(axis=1, keepdims=True), 1e-12)
    profile = {
        "recording": info,
        "frames": len(f0),
        "active_frames": int(data.active.sum()),
        "voiced_frames": int(voiced.sum()),
        "ste_dbfs": summarize(10 * np.log10(np.maximum(data.energy[data.active], 1e-12))),
        "zcr": summarize(data.zcr[data.active]),
        "pitch": {
            "median_hz": median,
            "percentiles_hz": summarize(pitch),
            "log2_scale": float((np.quantile(log_pitch, 0.9) - np.quantile(log_pitch, 0.1)) / 2.56),
            "confidence": summarize(confidence[voiced]),
            "cepstral_hz": summarize(cepstral[cepstral > 0]) if np.any(cepstral > 0) else None,
            "cepstral_agreement_within_10pct": (
                float(np.mean(np.abs(cepstral[agree] / f0[agree] - 1) <= 0.10)) if agree.any() else None
            ),
            "histogram": dict(zip(("counts", "edges_hz"),
                                  (v.tolist() for v in np.histogram(pitch, bins=28,
                                                                   range=(cfg.pitch_min, cfg.pitch_max))))),
        },
        "harmonic_relative_median_amplitudes": np.median(relative, axis=0).tolist(),
        "band_power_examples": spectral_examples(data, voiced, cfg),
        "lpc": envelope_bank(data.samples, confidence, cfg),
    }
    if include_cepstral:
        profile["cepstral_2ms_power_examples"] = spectral_examples(data, voiced, cfg, cepstral=True)
    return profile, (x, data, f0, cepstral, confidence)


def create_profiles(sources: tuple[Path, Path], path: Path, cfg: Settings, *,
                    include_cepstral: bool = False) -> tuple[dict, dict]:
    profiles = {"version": 1, "settings": asdict(cfg)}
    diagnostics = {}
    for speaker, source in zip(SPEAKERS, sources):
        x, _, info = partition(source, cfg)
        profiles[speaker], diagnostics[speaker] = calibrate_speaker(x, info, cfg, include_cepstral=include_cepstral)
    write_json(path, profiles)
    return profiles, diagnostics


def load_profiles(path: Path) -> tuple[dict, Settings]:
    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
        if profiles["version"] != 1:
            raise ValueError("unsupported version")
        cfg = Settings(**profiles["settings"])
        for speaker in SPEAKERS:
            profile = profiles[speaker]
            for field in ("band_power_examples", "cepstral_2ms_power_examples"):
                if field not in profile:
                    continue
                bank = np.asarray(profile[field], dtype=float)
                if (bank.ndim != 2 or not len(bank) or bank.shape[1] != len(band_starts(cfg))
                        or not np.isfinite(bank).all() or np.any(bank < 0)
                        or not np.allclose(bank.sum(axis=1), 1)):
                    raise ValueError(f"invalid {speaker} band powers")
            pitch = profile["pitch"]
            if not cfg.pitch_min <= pitch["median_hz"] <= cfg.pitch_max:
                raise ValueError(f"invalid {speaker} median pitch")
            if not np.isfinite(pitch["log2_scale"]) or pitch["log2_scale"] < 0:
                raise ValueError(f"invalid {speaker} pitch spread")
            bank = np.asarray(profile["lpc"]["log_envelopes"], dtype=float)
            if bank.ndim != 2 or not len(bank) or bank.shape[1] != cfg.fft // 2 + 1 or not np.isfinite(bank).all():
                raise ValueError(f"invalid {speaker} LPC envelopes")
            recording = profile["recording"]
            if len(recording["sha256"]) != 64 or recording["processing_rate_hz"] != cfg.rate:
                raise ValueError(f"invalid {speaker} source provenance")
            start, end = recording["calibration_samples"]
            held_start, held_end = recording["held_out_samples"]
            if not 0 <= start < end <= held_start < held_end:
                raise ValueError(f"invalid {speaker} partition")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"{path}: missing or incompatible calibration profiles ({exc}); run calibrate") from exc
    return profiles, cfg

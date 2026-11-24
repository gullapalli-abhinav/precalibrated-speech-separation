"""Local enrollment uses the same calibration features and 40 ms analysis as the offline release."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from config import DEFAULT, SPEAKERS
from src.calibration import band_starts, calibrate_speaker, load_profiles
from src.pitch_tracking import frames, single_pitch


def enroll(x, *, expected_seconds=None, clipped_fraction=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < DEFAULT.rate*2:
        raise ValueError("Supply at least two seconds of finite mono speech")
    seconds = len(x)/DEFAULT.rate
    if expected_seconds is not None and abs(seconds-expected_seconds) > .05:
        raise ValueError("Recording ended before the selected duration")
    if np.mean(abs(x) >= .995) > .002 or (clipped_fraction or 0) > .002:
        raise ValueError("Clipping detected. Reduce microphone gain and retry")
    data = frames(x, DEFAULT)
    active = data.energy > max(10**(-50/10), data.energy.max()*1e-4)
    if active.mean() < .20 or active.sum()*DEFAULT.hop/DEFAULT.rate < 1:
        raise ValueError("Insufficient active speech or input too quiet. Check the input level and retry")
    pitch, _, _ = single_pitch(data, DEFAULT)
    if np.count_nonzero((pitch > 0) & active) < max(30, .07*len(active)):
        raise ValueError("Too few reliable voiced frames. Record the speaker alone and retry")
    profile, _ = calibrate_speaker(x, {"file":"local enrollment", "duration_seconds":seconds},
                                   DEFAULT, include_cepstral=True)
    return profile


def session_profiles(profiles):
    return {"version":"live-1", "settings":asdict(DEFAULT), **{s:profiles[s] for s in SPEAKERS}}


def read_session(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("version") == 1:
        data, cfg = load_profiles(Path(path))
        if (cfg.window, cfg.hop, cfg.fft) != (DEFAULT.window, DEFAULT.hop, DEFAULT.fft):
            raise ValueError("Use a 40 ms, 10 ms hop, 1024 FFT profile")
    elif data.get("version") != "live-1" or data.get("settings") != asdict(DEFAULT):
        raise ValueError("Incompatible live profile format or settings")
    for s in SPEAKERS:
        profile = data[s]
        for key, columns in (("band_power_examples",len(band_starts(DEFAULT))),
                             ("cepstral_2ms_power_examples",len(band_starts(DEFAULT))),
                             ("lpc",DEFAULT.fft//2+1)):
            if key.startswith("cepstral") and key not in profile:
                continue
            bank = np.asarray(profile[key]["log_envelopes"] if key=="lpc" else profile[key], float)
            if (bank.ndim != 2 or not len(bank) or bank.shape[1] != columns or not np.isfinite(bank).all()
                    or (key != "lpc" and (np.any(bank < 0) or not np.allclose(bank.sum(1),1)))):
                raise ValueError(f"Invalid {s} {key} exemplars")
        pitch = profile["pitch"]
        if not (DEFAULT.pitch_min <= pitch["median_hz"] <= DEFAULT.pitch_max
                and np.isfinite(pitch["log2_scale"]) and pitch["log2_scale"] >= 0
                and np.isfinite(profile["ste_dbfs"]["p95"])):
            raise ValueError(f"Invalid {s} pitch or energy statistics")
    return {s:data[s] for s in SPEAKERS}

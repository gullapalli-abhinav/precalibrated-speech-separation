"""LPC envelopes and formant diagnostics from isolated calibration frames."""

import numpy as np
from scipy import fft, linalg

from config import Settings
from src.pitch_tracking import hann


def fit_lpc(frame: np.ndarray, cfg: Settings) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x = np.diff(frame, prepend=frame[0]) + 0.03 * np.r_[0, frame[:-1]]
    x = (x - x.mean()) * hann(cfg)
    correlation = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + cfg.lpc_order]
    correlation[0] += max(correlation[0] * 1e-3, 1e-12)
    a = np.r_[1.0, linalg.solve_toeplitz(correlation[:-1], -correlation[1:])]
    log_envelope = -np.log(np.maximum(np.abs(fft.rfft(a, cfg.fft)), 1e-9))
    log_envelope -= log_envelope.mean()
    poles = np.roots(a)
    frequency = np.angle(poles) * cfg.rate / (2 * np.pi)
    bandwidth = -np.log(np.maximum(np.abs(poles), 1e-12)) * cfg.rate / np.pi
    formants = np.sort(frequency[(frequency >= 150) & (frequency <= 4500) & (bandwidth < 450)])
    return a, log_envelope, formants.tolist()


def envelope_bank(samples: np.ndarray, confidence: np.ndarray, cfg: Settings) -> dict:
    indices = np.flatnonzero(confidence >= 0.65)
    if not len(indices):
        indices = np.flatnonzero(confidence)
    indices = indices[np.linspace(0, len(indices) - 1, min(24, len(indices)), dtype=int)]
    coefficients, envelopes, formants = zip(*(fit_lpc(samples[i], cfg) for i in indices))
    return {
        "frame_indices": indices.tolist(),
        "coefficients": np.asarray(coefficients).tolist(),
        "log_envelopes": np.asarray(envelopes).tolist(),
        "formants_hz": list(formants),
    }

"""Low-quefrency log-magnitude envelopes on the original real FFT grid."""

import numpy as np
from scipy import fft

from config import Settings


def real_cepstrum(spectrum: np.ndarray, cfg: Settings) -> np.ndarray:
    if spectrum.shape[-1] != cfg.fft // 2 + 1:
        raise ValueError("Spectrum does not match the configured real FFT grid")
    return fft.irfft(np.log(np.abs(spectrum) + 1e-12), n=cfg.fft, axis=-1)


def log_envelope(spectrum: np.ndarray, cfg: Settings, cutoff_ms: float) -> np.ndarray:
    """Retain both positive and negative quefrencies within the cutoff."""
    if not 0 <= cutoff_ms <= 1000*(cfg.fft//2)/cfg.rate:
        raise ValueError("Invalid quefrency cutoff")
    cep = real_cepstrum(spectrum, cfg)
    index = np.arange(cfg.fft)
    quefrency = np.minimum(index, cfg.fft-index)
    keep = quefrency <= round(cutoff_ms*cfg.rate/1000)
    return fft.rfft(cep*keep, n=cfg.fft, axis=-1).real

"""Pitch evidence and offline dual-speaker dynamic programming."""

from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage, signal

from config import SPEAKERS, Settings

EPS = 1e-12


@dataclass
class Frames:
    samples: np.ndarray
    spectrum: np.ndarray
    energy: np.ndarray
    zcr: np.ndarray
    active: np.ndarray


def hann(cfg: Settings) -> np.ndarray:
    return signal.windows.hann(cfg.window, sym=False)


def frames(x: np.ndarray, cfg: Settings) -> Frames:
    extra = (-len(x)) % cfg.hop
    padded = np.pad(x, (cfg.window // 2, cfg.window // 2 + extra))
    samples = np.lib.stride_tricks.sliding_window_view(padded, cfg.window)[::cfg.hop]
    spectrum = fft.rfft(samples * hann(cfg), n=cfg.fft)
    energy = np.mean(samples ** 2, axis=1)
    zcr = np.mean(np.diff(np.signbit(samples), axis=1), axis=1)
    # Combine an absolute floor with a relative floor for recording gain changes.
    threshold = max(1e-10, energy.max() * 10 ** (cfg.quiet_db / 10))
    return Frames(samples, spectrum, energy, zcr, energy > threshold)


def inverse_stft(spectrum: np.ndarray, length: int, cfg: Settings) -> np.ndarray:
    window = hann(cfg)
    synthesis = fft.irfft(spectrum, n=cfg.fft)[:, :cfg.window] * window
    output = np.zeros((len(spectrum) - 1) * cfg.hop + cfg.window)
    weight = np.zeros_like(output)
    square = window ** 2
    for i, sample in enumerate(synthesis):
        region = slice(i * cfg.hop, i * cfg.hop + cfg.window)
        output[region] += sample
        weight[region] += square
    start = cfg.window // 2
    region = slice(start, start + length)
    return output[region] / np.maximum(weight[region], EPS)


def periodicity(data: Frames, cfg: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Normalized correlation over overlapping samples and real cepstral peaks."""
    x = data.samples - data.samples.mean(axis=1, keepdims=True)
    size = fft.next_fast_len(2 * cfg.window - 1)
    ac = fft.irfft(np.abs(fft.rfft(x, n=size)) ** 2, n=size)
    end = int(np.ceil(cfg.rate / cfg.pitch_min)) + 2
    lag = np.arange(end)
    sums = np.pad(np.cumsum(x * x, axis=1), ((0, 0), (1, 0)))
    denominator = np.sqrt(np.maximum(sums[:, cfg.window - lag] * (sums[:, -1:] - sums[:, lag]), 0))
    ac = np.clip(ac[:, :end] / np.maximum(denominator, EPS), -1, 1)
    magnitude = np.abs(data.spectrum)
    cep = fft.irfft(np.log(np.maximum(magnitude, magnitude.max(axis=1, keepdims=True) * 1e-4 + EPS)),
                    n=cfg.fft)[:, :end]
    # Local prominence suppresses broad low-quefrency contributions.
    cep -= ndimage.gaussian_filter1d(cep, 4, axis=1)
    lo = int(cfg.rate / cfg.pitch_max)
    cep = np.clip(cep / np.maximum(np.max(cep[:, lo:], axis=1, keepdims=True), 0.035), 0, 1)
    return ac, cep


def pitch_grid(cfg: Settings) -> np.ndarray:
    return np.unique(np.r_[np.arange(cfg.pitch_min, cfg.pitch_max, cfg.pitch_step), cfg.pitch_max])


def interpolate(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.clip(positions, 0, values.shape[1] - 1)
    lower = positions.astype(int)
    upper = np.minimum(lower + 1, values.shape[1] - 1)
    frac = positions - lower
    return values[:, lower] * (1 - frac) + values[:, upper] * frac


def harmonic_grid(pitch: np.ndarray, freq: np.ndarray, bin_hz: float) -> np.ndarray:
    """A smoothly widening comb on the actual FFT frequency grid."""
    safe = np.maximum(pitch[..., None], 1)
    harmonic = np.maximum(np.rint(freq / safe), 1)
    width = np.minimum(1.10 * bin_hz + 0.009 * freq, 0.24 * safe)
    likelihood = np.exp(-0.5 * ((freq - harmonic * safe) / width) ** 2)
    return likelihood * (pitch[..., None] > 0)


def spectral_evidence(data: Frames, cfg: Settings, *, refined: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = 2 if refined else 1
    freq = fft.rfftfreq(resolution * cfg.fft, 1 / cfg.rate)
    grid = pitch_grid(cfg)
    band = (freq >= cfg.pitch_min - 15) & (freq <= 3500)
    # Score local spectral peaks so a second voice can be detected between the
    # dominant voice's harmonics. Synthesis uses a wider, independent comb.
    magnitude = np.abs(fft.rfft(data.samples * hann(cfg), n=resolution * cfg.fft) if refined else data.spectrum)
    level = ndimage.gaussian_filter1d(magnitude, 7 * resolution, axis=1)
    whitened = magnitude / (level + (0.025 if refined else 0.06) * magnitude.max(axis=1, keepdims=True) + EPS)
    maxima = (magnitude == ndimage.maximum_filter1d(magnitude, 3, axis=1))
    maxima &= magnitude >= 0.015 * magnitude.max(axis=1, keepdims=True)
    peaks = (whitened[:, band] * np.sqrt(magnitude[:, band]) * maxima[:, band]).astype(np.float32)
    peaks /= np.maximum(peaks.sum(axis=1, keepdims=True), EPS)
    f = freq[band]
    pitch = grid[:, None]
    nearest = np.maximum(np.rint(f / pitch), 1) * pitch
    width = np.minimum(0.77 * cfg.rate / cfg.fft + 0.004 * f, 0.18 * pitch)
    comb = np.exp(-0.5 * ((f - nearest) / width) ** 2).astype(np.float32)
    # A dense comb covers more spectral bins by chance. Subtract that coverage.
    harmonic = peaks @ comb.T - comb.mean(axis=1)
    return harmonic, peaks, comb


def single_pitch(data: Frames, cfg: Settings) -> tuple[np.ndarray, ...]:
    """Independent autocorrelation and cepstral estimates with a voiced gate."""
    ac, cep = periodicity(data, cfg)
    lag = np.arange(int(np.ceil(cfg.rate / cfg.pitch_max)), int(cfg.rate / cfg.pitch_min) + 1)
    local = (ac[:, lag] > ac[:, lag - 1]) & (ac[:, lag] >= ac[:, lag + 1])
    # Preferring the earliest nearly equivalent peak avoids common subharmonics.
    strength = np.where(local, ac[:, lag], -1)
    best = strength.max(axis=1)
    plausible = local & (strength >= best[:, None] - 0.13)
    chosen = lag[np.argmax(plausible, axis=1)]
    index = np.arange(len(chosen))
    middle = ac[index, chosen]
    left, right = ac[index, chosen - 1], ac[index, chosen + 1]
    offset = np.clip(0.5 * (right - left) / np.maximum(2 * middle - left - right, EPS), -0.5, 0.5)
    voiced = data.active & plausible.any(axis=1) & (middle >= 0.52) & (data.zcr < cfg.unvoiced_zcr)
    cep_lag = lag[np.argmax(cep[:, lag] - 0.10 * lag / lag[-1], axis=1)]
    return (
        np.where(voiced, np.clip(cfg.rate / (chosen + offset), cfg.pitch_min, cfg.pitch_max), 0),
        np.where(data.active & (cep[index, cep_lag] >= 0.45), cfg.rate / cep_lag, 0),
        np.where(voiced, middle, 0),
    )


@dataclass
class Tracks:
    f0: np.ndarray
    confidence: np.ndarray


def track(data: Frames, profiles: dict, cfg: Settings, *, legacy: bool = False,
          transition_scale: float | None = None, refined: bool | None = None, diagnostics: dict | None = None) -> Tracks:
    grid = pitch_grid(cfg)
    ac, cep = periodicity(data, cfg)
    harmonic, peaks, comb = spectral_evidence(data, cfg, refined=not legacy if refined is None else refined)
    lag = cfg.rate / grid
    periodic = 0.10 * np.maximum(interpolate(ac, lag), 0) + 0.07 * interpolate(cep, lag)
    zcr_penalty = 0.24 * np.clip((data.zcr - cfg.unvoiced_zcr) / 0.15, 0, 1)
    priors = []
    for speaker in SPEAKERS:
        pitch = profiles[speaker]["pitch"]
        scale = max(pitch["log2_scale"], 0.13)
        distance = np.abs(np.log2(grid / pitch["median_hz"])) / scale
        priors.append(np.minimum(0.055 * distance ** 2, 0.65))
    priors = np.array(priors)
    scores = harmonic[:, None, :] + periodic[:, None, :] - 0.31 - zcr_penalty[:, None, None] - priors
    # Discount peaks explained by the other voice. This operates on spectral
    # evidence from the mixture, never on an isolated reference signal.
    remaining = 1 - np.clip((comb - 0.08) / 0.55, 0, 1)
    centered = comb - comb.mean(axis=1, keepdims=True)
    secondary_base = 0.15 * periodic[:, None, :] - 0.022 - zcr_penalty[:, None, None] - priors
    best_other = scores[:, ::-1].argmax(axis=-1)
    residual = peaks[:, None, :] * remaining[best_other]
    def residual_score(mass: np.ndarray, base: np.ndarray) -> np.ndarray:
        if legacy:
            return base - 0.16 * np.sqrt(mass)
        # Spectral support is an integral over the *remaining* peak mass.
        # Scale the prior and voicing costs in the same units as that support.
        return mass * (base - 0.09) - 0.012

    secondary = residual @ centered.T + residual_score(residual.sum(axis=-1, keepdims=True), secondary_base)

    def proposals(values: np.ndarray) -> np.ndarray:
        local = values == ndimage.maximum_filter1d(values, size=5, axis=-1, mode="nearest")
        ranked = np.where(local, values, -np.inf)
        return np.argsort(ranked, axis=-1)[..., -min(3, len(grid)):]

    choices = np.concatenate((proposals(scores), proposals(secondary)), axis=-1)
    count = choices.shape[-1]
    primary = np.take_along_axis(scores, choices, axis=-1)
    primary = np.where(data.active[:, None, None], primary, -np.inf)
    options = np.concatenate((grid[choices], np.zeros((*choices.shape[:2], 1))), axis=2)
    emissions = np.pad(primary, ((0, 0), (0, 0), (0, 1)))
    state_count = (count + 1) ** 2
    states = np.stack(np.broadcast_arrays(options[:, 0, :, None], options[:, 1, None, :]), axis=-1)
    states = states.reshape(-1, state_count, 2)
    frame_cost = emissions[:, 0, :, None] + emissions[:, 1, None, :]
    possible = np.flatnonzero(data.active)
    for i in possible:
        candidates = choices[i]
        leftover = remaining[candidates] * peaks[i]
        mass = leftover.sum(axis=-1)
        male_second = leftover[0] @ centered[candidates[1]].T
        male_second += residual_score(mass[0, :, None], secondary_base[i, 1, candidates[1]][None, :])
        female_second = centered[candidates[0]] @ leftover[1].T
        female_second += residual_score(mass[1, None, :], secondary_base[i, 0, candidates[0]][:, None])
        pair = np.maximum(primary[i, 0, :, None] + male_second, primary[i, 1, None, :] + female_second)
        distance = np.abs(np.log2(options[i, 0, :count, None] / options[i, 1, None, :count]))
        frame_cost[i, :count, :count] = pair - 0.23 * np.exp(-0.5 * (distance / 0.09) ** 2)

    emission = frame_cost.reshape(-1, state_count)
    log_pitch = np.log2(np.maximum(states, 1))
    present = states > 0
    previous = emission[0].copy()
    parents = np.zeros(emission.shape, dtype=np.int16)
    indices = np.arange(state_count)
    if transition_scale is None:
        transition_scale = 1.0 if legacy else 0.35
    for i in range(1, len(states)):
        delta = np.abs(log_pitch[i, :, None] - log_pitch[i - 1, None, :])
        both = present[i, :, None] & present[i - 1, None, :]
        change = present[i, :, None] ^ present[i - 1, None, :]
        transition = np.where(both, np.minimum(1.1 * delta + 5 * delta ** 2, 2), 0.14 * change)
        accumulated = previous[None, :] - transition_scale * transition.sum(axis=-1)
        parents[i] = accumulated.argmax(axis=1)
        previous = emission[i] + accumulated[indices, parents[i]]
        previous -= previous.max()
    path = np.empty(len(states), dtype=int)
    path[-1] = previous.argmax()
    for i in range(len(states) - 1, 0, -1):
        path[i - 1] = parents[i, path[i]]
    selected = states[np.arange(len(states)), path]
    # Compare the selected state to removing each voice with the other fixed.
    without = np.column_stack((count * (count + 1) + path % (count + 1),
                               path // (count + 1) * (count + 1) + count))
    evidence = emission[np.arange(len(states)), path, None] - np.take_along_axis(emission, without, axis=1)
    confidence = np.where(selected > 0, np.clip((evidence + 0.03) / 0.34, 0, 1), 0)
    if diagnostics is not None:
        diagnostics.update(grid=grid, harmonic=harmonic, scores=scores, choices=grid[choices],
                           frame_best=states[np.arange(len(states)), emission.argmax(axis=1)])
    return Tracks(selected, confidence)

"""Mixture-only inference and complementary time-frequency masks."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import fft, ndimage

from config import SPEAKERS, Settings
from src.audio_io import write_audio
from src.calibration import band_starts
from src.cepstrum import log_envelope
from src.pitch_tracking import EPS, Frames, Tracks, frames, harmonic_grid, inverse_stft, track


@dataclass
class Analysis:
    frames: Frames
    tracks: Tracks
    likelihood: np.ndarray
    seconds: float


@dataclass
class Separation:
    audio: np.ndarray
    masks: np.ndarray
    seconds: float


def analyze(x: np.ndarray, profiles: dict, cfg: Settings) -> Analysis:
    start = perf_counter()
    data = frames(x, cfg)
    tracks = track(data, profiles, cfg)
    frequency = fft.rfftfreq(cfg.fft, 1 / cfg.rate)
    likelihood = harmonic_grid(tracks.f0, frequency, cfg.rate / cfg.fft)
    return Analysis(data, tracks, likelihood, perf_counter() - start)


def lpc_evidence(analysis: Analysis, profiles: dict, cfg: Settings) -> np.ndarray:
    magnitude = np.abs(analysis.frames.spectrum)
    # Match broad local spectral shapes to several calibration exemplars.
    smooth = ndimage.gaussian_filter1d(magnitude, sigma=6, axis=1)
    frequency = fft.rfftfreq(cfg.fft, 1 / cfg.rate)
    band = (frequency >= 200) & (frequency <= 4500)
    observed = np.log(np.maximum(smooth[:, band], smooth.max(axis=1, keepdims=True) * 1e-3 + EPS))
    # Correct the slope from LPC pre-emphasis before comparing shapes.
    observed += np.log(np.maximum(2 * np.sin(np.pi * frequency[band] / cfg.rate), 0.03))
    observed -= observed.mean(axis=1, keepdims=True)
    observed /= np.maximum(np.linalg.norm(observed, axis=1, keepdims=True), EPS)
    weights = []
    for speaker in SPEAKERS:
        bank = np.asarray(profiles[speaker]["lpc"]["log_envelopes"])
        shapes = bank[:, band].copy()
        shapes -= shapes.mean(axis=1, keepdims=True)
        shapes /= np.maximum(np.linalg.norm(shapes, axis=1, keepdims=True), EPS)
        similarity = np.maximum(observed @ shapes.T, 0)
        best = np.argpartition(similarity, -3, axis=1)[:, -3:] if bank.shape[0] >= 3 else np.broadcast_to(
            np.arange(len(bank)), similarity.shape)
        matched = bank[best].mean(axis=1)
        matched -= np.median(matched, axis=1, keepdims=True)
        weights.append(np.clip(matched, -2, 2))
    return np.exp(0.22 * np.stack(weights, axis=1) * analysis.tracks.confidence[..., None])


def legacy_strengths(analysis: Analysis, cfg: Settings) -> tuple[np.ndarray, np.ndarray]:
    confidence = analysis.tracks.confidence
    recent = ndimage.gaussian_filter1d(confidence, 4, axis=0, mode="nearest")
    fallback = np.array([cfg.fallback_female, 1 - cfg.fallback_female])
    # Trust surrounding voiced activity weakly for unvoiced or ambiguous energy.
    background = 0.70 * fallback + 0.30 * (recent + fallback) / (1 + recent.sum(axis=1, keepdims=True))
    strengths = analysis.likelihood ** 1.7 * confidence[..., None]
    return strengths, 0.12 * background[..., None]


def match_band_powers(power: np.ndarray, banks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit one isolated envelope per voice to each mixture using nonnegative gains.

    For each pair the two-variable least-squares optimum is either interior
    or on a single-source axis. Keep the fit with the least weighted error.
    """
    outputs, fits = [], []
    for mix in np.array_split(power, max(1, (len(power) + 127) // 128)):
        scale = 1 / np.maximum(mix + 0.03 * mix.max(axis=1, keepdims=True), EPS)
        y = mix * scale
        a, b = [bank[None] * scale[:, None] for bank in banks]
        aa, bb = (np.sum(v*v, axis=2) for v in (a, b))
        ay, by = (np.sum(v*y[:, None], axis=2) for v in (a, b))
        ab = a @ b.transpose(0, 2, 1)
        aa, ay = aa[:, :, None], ay[:, :, None]
        bb, by = bb[:, None, :], by[:, None, :]
        det = aa*bb - ab**2
        # Near-collinear pairs use their better single-source boundary.
        interior = det > 1e-9 * aa*bb
        safe_det = np.where(interior, det, 1)
        ga = (ay*bb - by*ab) / safe_det
        gb = (by*aa - ay*ab) / safe_det
        interior &= (ga > 0) & (gb > 0)
        single_a = ay**2 / aa >= by**2 / bb
        ga = np.where(interior, ga, np.where(single_a, ay/aa, 0))
        gb = np.where(interior, gb, np.where(single_a, 0, by/bb))
        explained = ga*ay + gb*by
        row = np.arange(len(y))
        i, j = np.unravel_index(explained.reshape(len(y), -1).argmax(1), explained.shape[1:])
        outputs.append(np.stack((banks[0][i] * ga[row, i, j, None],
                                 banks[1][j] * gb[row, i, j, None]), axis=1))
        fits.append(np.clip(explained[row, i, j] / np.maximum((y*y).sum(1), EPS), 0, 1))
    return np.concatenate(outputs), np.concatenate(fits)


def power_strengths(analysis: Analysis, profiles: dict, cfg: Settings, conservative: bool, *,
                    comb_exponent: float = 1.7, confidence_scale: float = 1.,
                    band_time_sigma: float = 1.25) -> np.ndarray:
    starts = band_starts(cfg)
    n_bins = analysis.frames.spectrum.shape[1]
    widths = np.diff(np.r_[starts, n_bins])
    centers = starts + (widths - 1) / 2
    confidence = analysis.tracks.confidence[..., None] * confidence_scale
    comb = analysis.likelihood ** comb_exponent
    if all("band_power_examples" in profiles[s] for s in SPEAKERS):
        power = np.add.reduceat(np.abs(analysis.frames.spectrum)**2, starts, axis=1)
        bands, fit = match_band_powers(power, [np.asarray(profiles[s]["band_power_examples"]) for s in SPEAKERS])
        bands = ndimage.gaussian_filter1d(bands, band_time_sigma, axis=0, mode="nearest")
        # Interpolate density, since the bands have unequal widths.
        pos = np.interp(np.arange(n_bins), centers, np.arange(len(starts)))
        lo = pos.astype(int)
        hi = np.minimum(lo + 1, len(starts)-1)
        density = bands / widths
        envelope = density[..., lo] * (1 - (pos-lo)) + density[..., hi] * (pos-lo)
        if conservative:
            uncertainty = 0.02 + 0.25 * (1 - fit[:, None, None])
            envelope += uncertainty * envelope.mean(axis=1, keepdims=True)
        # Unit-average comb redistributes the envelope power among harmonics.
        modulation = comb / np.maximum(comb.mean(axis=2, keepdims=True), 0.05)
        strengths = envelope * (1 + confidence * (modulation - 1))
        return strengths + EPS
    # Profiles predating the spectral exemplars still carry pitch evidence.
    strengths, floor = legacy_strengths(analysis, cfg)
    return strengths + floor


MALE_VARIANTS = ("current", "power_only", "reduced_harmonic", "reduced_smoothing",
                 "conservative_mask", "power_ratio", "less_time", "less_frequency",
                 "half_confidence", "wider_comb", "less_band_smoothing", "presence_gain", "cepstral_assisted")


def soft_power_ratio(female: np.ndarray, male: np.ndarray, exponent: float) -> np.ndarray:
    """Bounded power-ratio mask with an equal share when both powers vanish."""
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("The power exponent must be finite and positive")
    scale = np.maximum(female, male)
    safe = np.where(scale > 0, scale, 1)
    f, m = (power / safe for power in (female, male))
    f, m = f ** exponent, m ** exponent
    return np.divide(m, f + m, out=np.full_like(m, .5), where=(f+m) > 0)


def separate(analysis: Analysis, length: int, profiles: dict, cfg: Settings, method: str,
             *, mask: str = "conservative", male_variant: str = "current",
             male_floor: float = .22, power_exponent: float = .75) -> Separation:
    start = perf_counter()
    if method not in ("harmonic", "harmonic_lpc"):
        raise ValueError(f"Unknown separation method: {method}")
    if mask == "legacy":
        strengths, floor = legacy_strengths(analysis, cfg)
        if method == "harmonic_lpc":
            strengths *= lpc_evidence(analysis, profiles, cfg)
        strengths += floor
    elif mask in ("power", "conservative"):
        strengths = power_strengths(analysis, profiles, cfg, mask == "conservative")
        if method == "harmonic_lpc":
            strengths *= lpc_evidence(analysis, profiles, cfg)
    else:
        raise ValueError(f"Unknown mask: {mask}")
    result = synthesize(analysis, strengths, length, cfg, start)
    if male_variant == "current":
        return result
    if male_variant not in MALE_VARIANTS or mask != "conservative":
        raise ValueError("Unknown male variant or incompatible mask")
    if not 0 <= male_floor <= 1:
        raise ValueError("The male floor must be between zero and one")
    smoothing = (.85, .55)
    if male_variant == "cepstral_assisted":
        key = "cepstral_2ms_power_examples"
        if not all(key in profiles[s] for s in SPEAKERS):
            raise ValueError("cepstral_assisted requires a profile made with calibrate --cepstral")
        starts = band_starts(cfg)
        n_bins = analysis.frames.spectrum.shape[1]
        centers = starts + (np.diff(np.r_[starts, n_bins])-1)/2
        power = np.exp(2*log_envelope(analysis.frames.spectrum, cfg, 2.))
        bands, fit = match_band_powers(np.add.reduceat(power, starts, axis=1),
                                       [np.asarray(profiles[s][key]) for s in SPEAKERS])
        ratio = soft_power_ratio(bands[:, 0], bands[:, 1], 1.)
        envelope_mask = np.stack([np.interp(np.arange(n_bins), centers, row) for row in ratio])
        original = result.masks[:, 1]
        male = original + .35*fit[:, None]*(envelope_mask-original)
    elif male_variant in ("conservative_mask", "presence_gain"):
        hz = fft.rfftfreq(cfg.fft, 1 / cfg.rate)
        low = (hz >= 80) & (hz < 1000)
        energy = np.abs(analysis.frames.spectrum[:, low])**2
        evidence = (energy * result.masks[:, 1, low]).sum(1) / np.maximum(energy.sum(1), EPS)
        support = ndimage.gaussian_filter1d(evidence, 2, mode="nearest")
        ramp = np.clip((hz-300)/1500, 0, 1)
        original = result.masks[:, 1]
        if male_variant == "presence_gain":
            # Raise partially passed speech bands while keeping rejected bins quiet.
            male = original + .65*support[:, None]*ramp*np.minimum(original, 1-original)
        else:
            male = original + male_floor*support[:, None]*ramp*(1-original)
    else:
        if male_variant in ("power_only", "reduced_harmonic", "half_confidence", "wider_comb", "less_band_smoothing"):
            if not all("band_power_examples" in profiles[s] for s in SPEAKERS):
                raise ValueError("This male variant requires band-power examples. Run calibrate.")
            strengths = power_strengths(analysis, profiles, cfg, True,
                comb_exponent=.85 if male_variant in ("reduced_harmonic", "wider_comb") else 1.7,
                confidence_scale=0 if male_variant == "power_only" else
                    .5 if male_variant in ("reduced_harmonic", "half_confidence") else 1.,
                band_time_sigma=.4 if male_variant == "less_band_smoothing" else 1.25)
            if method == "harmonic_lpc" and male_variant != "power_only":
                strengths *= lpc_evidence(analysis, profiles, cfg)
        if male_variant in ("less_time", "less_frequency", "reduced_smoothing"):
            smoothing = (.25 if male_variant != "less_frequency" else .85,
                         .15 if male_variant != "less_time" else .55)
        male = soft_power_ratio(strengths[:, 0], strengths[:, 1],
                                power_exponent if male_variant == "power_ratio" else 1.)
        male = ndimage.gaussian_filter(male, sigma=smoothing, mode="nearest")
    result.masks[:, 1] = male
    result.audio[1] = inverse_stft(analysis.frames.spectrum * male, length, cfg)
    result.seconds = perf_counter() - start
    return result


def synthesize(analysis: Analysis, strengths: np.ndarray, length: int, cfg: Settings,
               start: float, smoothing: tuple[float, float] = (0.85, 0.55)) -> Separation:
    female = strengths[:, 0] / np.maximum(strengths.sum(axis=1), EPS)
    female = ndimage.gaussian_filter(female, sigma=smoothing, mode="nearest")
    masks = np.stack((female, 1 - female), axis=1)
    outputs = np.stack([inverse_stft(analysis.frames.spectrum * masks[:, i], length, cfg) for i in range(2)])
    return Separation(outputs, masks, perf_counter() - start)


def select(x: np.ndarray, separated: np.ndarray, mode: str) -> np.ndarray:
    return {"mute-female": separated[1], "mute-male": separated[0],
            "both": x, "mute-both": np.zeros_like(x)}[mode]


def save_outputs(directory: Path, x: np.ndarray, result: Separation, cfg: Settings, mode: str) -> None:
    for speaker, audio in zip(SPEAKERS, result.audio):
        write_audio(directory / f"output_{speaker}.wav", audio, cfg.rate)
    write_audio(directory / "output_mixture.wav", x, cfg.rate)
    write_audio(directory / "output_selected.wav", select(x, result.audio, mode), cfg.rate)

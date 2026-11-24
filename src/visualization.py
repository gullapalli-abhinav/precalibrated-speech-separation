"""Headless diagnostics using physical time and frequency axes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SPEAKERS, Settings
from src.pitch_tracking import Frames, frames
from src.separator import Analysis, Separation

COLORS = ("#943b79", "#1a738d")


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=135)
    plt.close(figure)


def waveform(ax: plt.Axes, x: np.ndarray, cfg: Settings, color: str) -> None:
    step = max(1, len(x) // 5000)
    starts = np.arange(0, len(x), step)
    time = starts / cfg.rate
    if step == 1:
        ax.plot(time, x, color=color, lw=0.6)
    else:
        ax.fill_between(time, np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts),
                        color=color, linewidth=0)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", xlim=(0, len(x) / cfg.rate))


def heat(ax: plt.Axes, values: np.ndarray, length: int, cfg: Settings, **kwargs) -> None:
    step = max(1, len(values) // 1400)
    limit = min(values.shape[1], int(4500 / (cfg.rate / cfg.fft)) + 1)
    bin_hz = cfg.rate / cfg.fft
    frame_s = step * cfg.hop / cfg.rate
    shown = values[::step, :limit]
    ax.imshow(shown.T, extent=(-frame_s / 2, (len(shown) - 0.5) * frame_s,
                              -bin_hz / 2, (limit - 0.5) * bin_hz),
              aspect="auto", origin="lower", interpolation="nearest", **kwargs)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", xlim=(0, length / cfg.rate),
           ylim=(0, (limit - 0.5) * bin_hz))


def spec(data: Frames, cfg: Settings) -> np.ndarray:
    # Hann coherent gain converts magnitudes to a sinusoid-amplitude scale.
    return 20 * np.log10(np.maximum(np.abs(data.spectrum) / (cfg.window / 4), 1e-7))


def calibration_plots(directory: Path, profiles: dict, diagnostics: dict, cfg: Settings) -> None:
    for name, color in zip(SPEAKERS, COLORS):
        x, data, f0, cep, confidence = diagnostics[name]
        time = np.arange(len(f0)) * cfg.hop / cfg.rate
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout="constrained")
        fig.suptitle(f"{name.capitalize()} calibration segment")
        waveform(axes[0, 0], x, cfg, color)
        heat(axes[0, 1], spec(data, cfg), len(x), cfg, cmap="magma", vmin=-70, vmax=-10)
        axes[0, 0].set_title("Waveform")
        axes[0, 1].set_title("Spectrogram (dBFS)")
        ax = axes[1, 0]
        ax.scatter(time[cep > 0], cep[cep > 0], s=3, color="#9da0a8", label="Real cepstrum")
        ax.scatter(time[f0 > 0], f0[f0 > 0], s=4, c=confidence[f0 > 0], cmap="viridis",
                   vmin=0.5, vmax=1, label="Autocorrelation")
        ax.legend(loc="upper right")
        ax.set(xlabel="Time (s)", ylabel="F0 (Hz)", title="Pitch and autocorrelation confidence",
               xlim=(0, len(x) / cfg.rate), ylim=(cfg.pitch_min - 5, cfg.pitch_max + 5))
        ax = axes[1, 1]
        ax.hist(f0[f0 > 0], bins=np.arange(cfg.pitch_min, cfg.pitch_max + 11, 10), color=color)
        ax.axvline(profiles[name]["pitch"]["median_hz"], color="#202630", linestyle="--", label="Median")
        ax.set(xlabel="F0 (Hz)", ylabel="Voiced frames", title="Pitch distribution")
        ax.legend()
        ax = axes[2, 0]
        freq = np.fft.rfftfreq(cfg.fft, 1 / cfg.rate)
        bank = np.asarray(profiles[name]["lpc"]["log_envelopes"])
        pick = np.linspace(0, len(bank) - 1, min(len(bank), 6), dtype=int)
        ax.plot(freq, (20 / np.log(10)) * bank[pick].T, alpha=0.65, lw=1)
        ax.set(xlabel="Frequency (Hz)", ylabel="Relative level (dB)", title="Representative LPC envelopes", xlim=(0, 4500))
        ax = axes[2, 1]
        ax.plot(time, 10 * np.log10(np.maximum(data.energy, 1e-12)), color=color, lw=0.6)
        ax.set(xlabel="Time (s)", ylabel="STE (dBFS)", title="Short-time energy and zero-crossing rate",
               xlim=(0, len(x) / cfg.rate))
        twin = ax.twinx()
        twin.plot(time, data.zcr, color="#657985", lw=0.6, alpha=0.6)
        twin.set_ylabel("ZCR")
        save(fig, directory / f"{name}_calibration.png")


def separation_plots(directory: Path, x: np.ndarray, analysis: Analysis, result: Separation,
                     cfg: Settings, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), layout="constrained")
    fig.suptitle(title)
    waveform(axes[0], x, cfg, "#263648")
    axes[0].set_title("Mixture waveform")
    heat(axes[1], spec(analysis.frames, cfg), len(x), cfg, cmap="magma", vmin=-70, vmax=-10)
    axes[1].set_title("Mixture spectrogram (dBFS)")
    time = np.arange(len(analysis.tracks.f0)) * cfg.hop / cfg.rate
    for i, (name, color) in enumerate(zip(SPEAKERS, COLORS)):
        f0 = analysis.tracks.f0[:, i]
        voiced = f0 > 0
        axes[2].scatter(time[voiced], f0[voiced], s=4 + 10 * analysis.tracks.confidence[voiced, i],
                        color=color, label=name.capitalize(), alpha=0.65)
    axes[2].legend(loc="upper right")
    axes[2].set(xlabel="Time (s)", ylabel="F0 (Hz)", title="Tracked F0 (marker size reflects confidence)",
                xlim=(0, len(x) / cfg.rate), ylim=(cfg.pitch_min - 5, cfg.pitch_max + 5))
    save(fig, directory / "mixture_and_pitch.png")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle(title)
    for i, name in enumerate(SPEAKERS):
        heat(axes[0, i], analysis.likelihood[:, i], len(x), cfg, cmap="viridis", vmin=0, vmax=1)
        heat(axes[1, i], result.masks[:, i], len(x), cfg, cmap="viridis", vmin=0, vmax=1)
        heat(axes[2, i], spec(frames(result.audio[i], cfg), cfg), len(x), cfg, cmap="magma", vmin=-70, vmax=-10)
        for row, label in enumerate(("harmonic likelihood (0–1)", "soft mask (0–1)", "output spectrogram (dBFS)")):
            axes[row, i].set_title(f"{name.capitalize()} {label}")
    save(fig, directory / "masks_and_outputs.png")


def comparison_plot(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, name in zip(axes, SPEAKERS):
        for method, color in zip(("harmonic", "harmonic_lpc"), COLORS):
            selected = [row for row in rows if row["condition"].startswith("ratio_") and row["method"] == method
                        and row["subset"] == "all" and row["target"] == name]
            x = [int(row["condition"].removeprefix("ratio_").removesuffix("dB")) for row in selected]
            ax.plot(x, [row["si_sdri_db"] for row in selected], marker="o", label=method, color=color)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set(title=f"{name.capitalize()} target", xlabel="Female-to-male RMS ratio (dB)", ylabel="SI-SDR improvement (dB)")
        ax.set_xticks(x)
        ax.legend()
    save(fig, path)

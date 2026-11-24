"""Render EC4143 figures from a saved session and deterministic streaming replay."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import fft

from config import DEFAULT as CFG, SPEAKERS
from src.audio_io import read_audio
from src.enrollment import read_session
from src.pitch_tracking import Frames, frames, harmonic_grid, periodicity, single_pitch
from src.streaming import StreamingSeparator

COLORS = {"male": "#0072B2", "female": "#A33283", "mixture": "#303030", "selected": "#00835C"}
SIGNALS = ("mixture", "male", "female", "selected")
FLOOR = -90
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


class Replay(StreamingSeparator):
    """Observe the exact inputs and returned masks at the existing DSP boundary."""

    def __init__(self, profiles):
        super().__init__(profiles)
        self.trace = []

    def _mask(self, spectrum, pitch, confidence, bands, fit):
        masks = super()._mask(spectrum, pitch, confidence, bands, fit)
        self.trace.append((self.start + self.cfg.window // 2, pitch.copy(), confidence.copy(),
                           spectrum.copy(), bands.copy(), float(fit), masks.copy()))
        return masks


def load_turns(folder, length):
    with (folder / "turns.csv").open(newline="", encoding="utf-8-sig") as file:
        turns = [(int(row["sample"]), row["state"], row["route"], int(row["turn_zero_based"]))
                 for row in csv.DictReader(file)]
    if any(not 0 <= t[0] <= length for t in turns) or any(
            b[0] < a[0] for a, b in zip(turns, turns[1:])):
        raise ValueError("Turn log must contain ordered sample indices within the session")
    return turns


def mark_turns(ax, turns, duration, *, shade=True):
    for i, (sample, _, route, _) in enumerate(turns):
        start = sample / CFG.rate
        end = turns[i + 1][0] / CFG.rate if i + 1 < len(turns) else duration
        if shade:
            ax.axvspan(start, end, color=COLORS.get(route, "#777777"), alpha=.09, lw=0)
        if sample:
            ax.axvline(start, color=COLORS.get(route, "#444444") if shade else "white",
                       lw=.8, ls="--", alpha=.8)
    ax.set_xlim(0, duration)


def representative_calibration(path, median):
    x, _ = read_audio(path, CFG.rate)
    data = frames(x, CFG)
    f0, _, confidence = single_pitch(data, CFG)
    voiced = np.flatnonzero(f0 > 0)
    if not voiced.size:
        raise ValueError(f"No reliable calibration pitch in {path}")
    # Prefer a strong periodic frame near the enrolled median.
    cost = np.abs(np.log2(f0[voiced] / median)) + .15 * (1 - confidence[voiced])
    j = voiced[cost.argmin()]
    ac, _ = periodicity(Frames(*(field[j:j+1] for field in (
        data.samples, data.spectrum, data.energy, data.zcr, data.active))), CFG)
    return ac[0], f0[j], j * CFG.hop / CFG.rate, confidence[j]


def stacf_axis(ax, ac, estimates, title):
    lag = 1000 * np.arange(len(ac)) / CFG.rate
    ax.plot(lag, ac, color=COLORS["mixture"], lw=1.2)
    ax.axhline(0, color="0.65", lw=.6)
    ax.axvspan(1000 / CFG.pitch_max, 1000 / CFG.pitch_min, color="0.7", alpha=.13)
    for speaker, pitch in estimates.items():
        if pitch <= 0:
            continue
        period = 1000 / pitch
        ax.axvline(period, color=COLORS[speaker], ls="--", lw=1.2)
        ax.scatter(period, np.interp(period, lag, ac), color=COLORS[speaker], s=23, zorder=3,
                   label=f"{speaker.title()}  {period:.2f} ms | {pitch:.1f} Hz")
    if any(pitch > 0 for pitch in estimates.values()):
        ax.legend(fontsize=8, loc="lower right")
    ax.set(title=title, xlabel="Lag (ms)", ylabel="Normalized STACF", ylim=(-1.05, 1.05))


def save(fig, folder, name):
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def pitch_figure(folder, profiles, calibration, mixed_ac, centers, pitch, index, turns, duration):
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.35))
    for column, speaker in enumerate(("male", "female")):
        ax = fig.add_subplot(grid[0, column])
        if speaker in calibration:
            ac, f0, when, _ = calibration[speaker]
            stacf_axis(ax, ac, {speaker: f0}, f"{speaker.title()} calibration WAV · {when:.2f} s")
        else:
            ax.set_axis_off()
            ax.text(.5, .5, f"{speaker.title()} calibration waveform unavailable\n"
                    f"Supply --{speaker}-calibration to plot its STACF.",
                    ha="center", va="center", transform=ax.transAxes)
    stacf_axis(fig.add_subplot(grid[0, 2]), mixed_ac, dict(zip(SPEAKERS, pitch[index])),
               f"Mixture replay · {centers[index]:.2f} s")
    ax = fig.add_subplot(grid[1, :])
    mark_turns(ax, turns, duration)
    for i, speaker in enumerate(SPEAKERS):
        stats = profiles[speaker]["pitch"]["percentiles_hz"]
        color = COLORS[speaker]
        ax.axhspan(stats["p05"], stats["p95"], color=color, alpha=.10)
        ax.axhline(stats["median"], color=color, ls=":", lw=.9)
        ax.plot(centers, np.where(pitch[:, i] > 0, pitch[:, i], np.nan), color=color,
                lw=.75, marker=".", ms=1.6, label=f"{speaker.title()} F0 · replay")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=".35", ls="--", label="Logged route transition"))
    ax.legend(handles=handles, loc="upper right", ncol=3)
    ax.set(xlabel="Session time (s)", ylabel="Estimated F0 (Hz)", ylim=(60, 360),
           title="Joint F0 tracks · horizontal bands show enrollment P05–P95, dotted lines show medians")
    fig.suptitle("01  Short-time autocorrelation and fundamental frequency", fontweight="bold")
    save(fig, folder, "01_stacf_pitch")


def spectral_figure(folder, replay, indices, overlap):
    fig, axes = plt.subplots(4, len(indices), squeeze=False, sharex=True, sharey="row",
                             figsize=(6.6 * len(indices), 9), layout="constrained")
    for col, index in enumerate(indices):
        center, pitch, confidence, spectrum, bands, fit, masks = replay.trace[index]
        mixture, power, mask, separated = axes[:, col]
        freq = replay.freq
        magnitude = abs(spectrum)
        mixture.plot(freq, magnitude, color=COLORS["mixture"], lw=1)
        for i, speaker in enumerate(SPEAKERS):
            color = COLORS[speaker]
            if pitch[i] > 0:
                harmonics = np.arange(pitch[i], 4000, pitch[i])
                mixture.vlines(harmonics, 0, magnitude[freq <= 4000].max() * .12,
                               color=color, alpha=.55, lw=.75)
                mixture.axvline(pitch[i], color=color, ls="--", lw=1)
            mixture.plot([], [], color=color, label=f"{speaker.title()} {pitch[i]:.1f} Hz"
                         f" (confidence {confidence[i]:.2f})")
            density = np.interp(replay.pos, np.arange(len(replay.starts)), bands[i] / replay.widths)
            power.plot(freq, density, color=color, label=speaker.title())
            mask.plot(freq, masks[i], color=color)
            separated.plot(freq, abs(spectrum * masks[i]), color=color)
        label = ("Harmonic overlap" if col else
                 "Dual-pitch frame" if np.all(pitch > 0) else "Highest-energy frame")
        mixture.set_title(f"{label} · "
                          f"{center / CFG.rate:.2f} s · overlap score {overlap[index]:.2f}")
        mixture.legend(fontsize=8)
        power.text(.99, .92, f"Weighted band fit q = {fit:.2f}", ha="right", va="top",
                   transform=power.transAxes, fontsize=8)
        floor = max(float(np.max(bands / replay.widths)) * 1e-8, 1e-12)
        power.set_ylim(bottom=floor)
        power.set_yscale("log", nonpositive="clip")
        mask.set(ylim=(-.02, 1.02))
        separated.set(xlim=(0, 4000), xlabel="Frequency (Hz)")
    for ax, label in zip(axes[:, 0], ("Mixture |X|", "Fitted power / bin", "Applied mask",
                                     "Separated |Y|")):
        ax.set_ylabel(label)
    cep = ("2 ms cepstral correction active on male mask" if replay.cep_banks is not None
           else "Male mask complements female mask · cepstral mask correction inactive")
    fig.suptitle("02  STFT and spectral separation · replay\n"
                 f"Power curves precede the uncertainty floor, pitch comb and LPC weighting · {cep}",
                 fontsize=11, fontweight="bold")
    save(fig, folder, "02_stft_spectral_masks")


def comparative_figure(folder, audio, turns, duration):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(13, 10), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.2]})
    freq = fft.rfftfreq(CFG.fft, 1 / CFG.rate)
    ste_ax = axes[-1]
    for ax, name in zip(axes, SIGNALS):
        data = frames(audio[name], CFG)
        times = np.arange(len(data.energy)) * CFG.hop / CFG.rate
        amplitude = abs(data.spectrum) / (CFG.window / 2)
        amplitude[:, 1:-1] *= 2
        db = 20 * np.log10(np.maximum(amplitude, 10 ** (FLOOR / 20)))
        mesh = ax.pcolormesh(times, freq / 1000, db.T, shading="auto", cmap="magma",
                             vmin=FLOOR, vmax=0, rasterized=True)
        mark_turns(ax, turns, duration, shade=False)
        ax.set(ylabel=f"{name.title()}\nkHz", ylim=(0, 8), yticks=(0, 2, 4, 6, 8))
        ste_ax.plot(times, 10 * np.log10(np.maximum(data.energy, 10 ** (FLOOR / 10))),
                    label=name.title(), color=COLORS[name], lw=.9)
    fig.colorbar(mesh, ax=axes[:4], label="Magnitude (dBFS)", shrink=.9)
    mark_turns(ste_ax, turns, duration)
    ste_ax.legend(ncol=4, loc="lower right")
    ste_ax.set(xlabel="Session time (s)", ylabel="STE (dBFS)", ylim=(FLOOR, 0))
    fig.suptitle("03  Recorded spectrograms and short-time energy\n"
                 "Dashed lines mark logged transitions · blue/purple shading shows male/female authorization",
                 fontsize=12, fontweight="bold")
    save(fig, folder, "03_spectrogram_ste")

def analyze(session, profile_path, calibration_paths):
    profiles = read_session(profile_path)
    audio = {}
    rates = {}
    for name in SIGNALS:
        audio[name], info = read_audio(session / f"{name}.wav", rate=None)
        rates[name] = info["rate_hz"], info["channels"]
    length = len(audio["mixture"])
    if any(rates[name] != (CFG.rate, 1) or len(audio[name]) != length for name in SIGNALS):
        raise ValueError("Expected four aligned mono session WAVs at 16 kHz")
    turns = load_turns(session, length)
    duration = length / CFG.rate
    calibration = {s: representative_calibration(path, profiles[s]["pitch"]["median_hz"])
                   for s, path in calibration_paths.items() if path is not None}
    replay = Replay(profiles)
    mix = audio["mixture"].astype("float32")
    for start in range(0, length, 512):
        replay.feed(mix[start:start + 512])
    replay.finish()
    replay.trace = [row for row in replay.trace if row[0] < length]
    centers, pitch, confidence, spectra, fits = (np.asarray([row[k] for row in replay.trace])
                                                  for k in (0, 1, 2, 3, 5))
    dual = np.flatnonzero(np.all(pitch > 0, axis=1))
    magnitude = abs(spectra)
    energy = np.sum(magnitude**2, axis=1)
    # Rank observed dual-F0 frames by joint confidence and mixture spectral energy.
    score = confidence.min(1) * np.sqrt(energy / max(float(energy.max()), 1e-12))
    index = int(dual[np.argmax(score[dual])]) if len(dual) else int(energy.argmax())
    overlap = np.zeros(len(centers))
    if len(dual):
        hz = (replay.freq >= 70) & (replay.freq <= 4000)
        comb = harmonic_grid(pitch[dual], replay.freq[hz], CFG.rate / CFG.fft)
        shared = (comb[:, 0] >= .5) & (comb[:, 1] >= .5)
        power = magnitude[dual][:, hz]**2
        overlap[dual] = (power * shared).sum(1) / np.maximum(power.sum(1), 1e-12)
    indices = [index]
    eligible = dual[(overlap[dual] >= .20) & (score[dual] >= .25 * score[index]) &
                    (abs(centers[dual] - centers[index]) >= CFG.window)]
    if eligible.size:
        indices.append(int(eligible[overlap[eligible].argmax()]))
    # Reconstruct the exact float32 input frame for the replay STACF.
    padded = np.pad(mix, (CFG.window // 2, CFG.window // 2))
    x = padded[centers[index]:centers[index] + CFG.window]
    ac, _ = periodicity(Frames(x[None], spectra[index:index+1], None, None, None), CFG)

    folder = session / "analysis"
    folder.mkdir(exist_ok=True)
    pitch_figure(folder, profiles, calibration, ac[0], centers/CFG.rate, pitch, index, turns, duration)
    spectral_figure(folder, replay, indices, overlap)
    comparative_figure(folder, audio, turns, duration)

    rows = [("duration", "", duration, "s", "recorded WAV"),
            ("logged_route_transitions", "", sum(t[0] > 0 for t in turns), "count", "turn log"),
            ("dual_positive_f0_fraction", "", len(dual)/len(centers), "fraction", "replay")]
    energies = {name: float(np.dot(x, x)) for name, x in audio.items()}
    attenuation = (10 * np.log10(energies["mixture"] / energies["selected"])
                   if energies["selected"] and energies["mixture"] else None)
    for name, total in energies.items():
        rows += [("total_energy", name, total, "FS^2 samples", "recorded WAV"),
                 ("rms", name, np.sqrt(total / length), "FS", "recorded WAV")]
    rows.append(("mixture_to_selected_total_energy_attenuation", "", attenuation, "dB", "recorded WAV"))
    medians = {}
    for i, speaker in enumerate(SPEAKERS):
        positive = pitch[:, i] > 0
        medians[speaker] = float(np.median(pitch[positive, i])) if positive.any() else None
        stats = profiles[speaker]["pitch"]["percentiles_hz"]
        rows += [("enrollment_p05_f0", speaker, stats["p05"], "Hz", "profile"),
                 ("enrollment_p95_f0", speaker, stats["p95"], "Hz", "profile"),
                 ("median_positive_f0", speaker, medians[speaker], "Hz", "replay"),
                 ("positive_f0_fraction", speaker, positive.mean(), "fraction", "replay")]
        if speaker in calibration:
            _, f0, time, strength = calibration[speaker]
            rows += [("calibration_frame_center", speaker, time, "s", "calibration WAV"),
                     ("calibration_frame_f0", speaker, f0, "Hz", "calibration WAV"),
                     ("calibration_pitch_period", speaker, 1000/f0, "ms", "calibration WAV"),
                     ("calibration_pitch_lag", speaker, CFG.rate/f0, "samples", "calibration WAV"),
                     ("calibration_peak_stacf", speaker, strength, "correlation", "calibration WAV")]
    for number, j in enumerate(indices, 1):
        label = f"frame {number}"
        rows += [("frame_center", label, centers[j]/CFG.rate, "s", "replay"),
                 ("harmonic_overlap_power_fraction", label, overlap[j], "fraction", "replay"),
                 ("band_fit_quality", label, fits[j], "fraction", "replay")]
        for i, speaker in enumerate(SPEAKERS):
            rows += [("frame_f0", f"{label} {speaker}", pitch[j, i] or None, "Hz", "replay"),
                     ("pitch_confidence", f"{label} {speaker}", confidence[j, i], "score", "replay"),
                     ("frame_pitch_lag", f"{label} {speaker}",
                      CFG.rate/pitch[j, i] if pitch[j, i] else None, "samples", "replay"),
                     ("frame_pitch_period", f"{label} {speaker}",
                      1000/pitch[j, i] if pitch[j, i] else None, "ms", "replay")]
    for tick, state, route, turn in turns:
        rows.append(("route_event", f"{state}/{route}/turn {turn}", tick/CFG.rate, "s", "turn log"))

    with (folder / "dsp_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "signal_or_frame", "value", "unit", "source"))
        writer.writerows(rows)
    write_summary(folder, profile_path, calibration_paths, calibration, medians, profiles,
                  centers, pitch, fits, indices, overlap, replay.cep_banks is not None,
                  duration, turns, attenuation)
    return folder


def write_summary(folder, profile_path, calibration_paths, calibration, medians, profiles,
                  centers, pitch, fits, indices, overlap, cepstral, duration, turns, attenuation):
    def hz(value):
        return f"{value:.2f} Hz" if value else "no positive estimate"

    calibration_notes = []
    for speaker in ("male", "female"):
        stats = profiles[speaker]["pitch"]["percentiles_hz"]
        note = (f"- {speaker.title()} profile P05–P95 is {stats['p05']:.2f}–{stats['p95']:.2f} Hz. "
                f"Session median over positive estimates is {hz(medians[speaker])}.")
        if speaker in calibration:
            _, f0, when, _ = calibration[speaker]
            note += (f" The frame at {when:.2f} s in **{calibration_paths[speaker].name}** "
                     f"gives {hz(f0)} and {1000/f0:.2f} ms.")
        else:
            note += " Its raw calibration WAV was unavailable."
        calibration_notes.append(note)
    frame_notes = []
    for j in indices:
        f, m = pitch[j]
        frame_notes.append(
            f"- At {centers[j]/CFG.rate:.3f} s, male F0 is {hz(m)} and female F0 is {hz(f)}. "
            f"The weighted power fit is {fits[j]:.3f}. "
            f"Shared harmonic bins contain {overlap[j]:.1%} of mixture spectral power from 70 Hz to 4 kHz."
        )
    if len(indices) == 1:
        frame_notes.append("- No separate dual-F0 frame met the overlap and confidence criteria.")
    amp_note = (f"The measured total energy attenuation is **{attenuation:.2f} dB**."
                if attenuation is not None else "Total attenuation is undefined as a finite dB value when either energy is zero.")
    calibration_text = "\n".join(calibration_notes)
    frame_text = "\n".join(frame_notes)
    branch = ("both profiles supply cepstral banks, so the male correction contributes according to fit quality"
              if cepstral else "one or both profiles lack cepstral banks, so the male mask is complementary")
    (folder/"analysis_summary.md").write_text(f"""# EC4143 post-session DSP analysis

The session spans {duration:.2f} s. The four WAVs and {len(turns)} route-log rows are original session artifacts.
F0, weighted fits, power envelopes and masks are deterministic offline replay results using
**{profile_path.name}** and the saved 16 kHz PCM mixture. Original floating-point microphone samples
and runtime masks were not saved. Replay values may differ because the mixture was quantized or
clipped before storage. Figure 3 always reads the original selected WAV, including any intentional or device-era silence.

## Figure 1 · STACF and fundamental frequency

**Syllabus.** Short-time stationarity, normalized autocorrelation, pitch and voicing.
For the mean-removed frame, r[l] = sum(x[n] x[n+l]) / sqrt(sum(x[n]^2) sum(x[n+l]^2)),
with the sums restricted to overlapping samples. T0 = l / Fs and F0 = Fs / l.

**Interpretation.** The isolated plots use the actual calibration pitch detector on the supplied WAVs.
The profile ranges come from the saved enrollment statistics. Live enrollment does not retain its raw
capture, so supply the WAVs used for enrollment when available. Other isolated examples illustrate
the same detector and should be identified as such.

{calibration_text}

The mixed-frame markers indicate periods inferred from the joint replay tracks, even if a marker
does not fall on a local STACF maximum. F0 gaps have no positive pitch estimate. The tracker combines
STACF, cepstral evidence, harmonic peaks and profile priors across frames. Harmonics can create peaks
at multiples or fractions of a period. Two voices share lag evidence and unvoiced speech lacks reliable
periodicity. Pitch estimates cannot establish who truly spoke in each frame.

**Likely viva questions**

1. Why normalize by overlapping energy? It makes lag scores comparable across changing overlap and overall signal level.
2. Why might two peaks imply an octave error? A true period also repeats at twice that lag, and strong even harmonics can favor half the lag.
3. Why leave gaps in F0? Aperiodic consonants or weak evidence can make a voiced hypothesis unreliable.

## Figure 2 · STFT and spectral masking

**Syllabus.** Windowed spectral analysis, harmonic structure, weighted least squares and time–frequency filtering.
X[t,k] = FFT(x[t]w)[k], band power P[b] = sum(|X[t,k]|^2 for k in b), and
Y_s[t,k] = M_s[t,k] X[t,k]. The fitted speaker curves show temporally smoothed fitted band powers
divided by their respective bin counts and interpolated between band centers.

**Interpretation.**

{frame_text}

The harmonic ticks show multiples of tracked F0. The paired mask curves are the exact masks
returned by the streaming DSP during replay. The power curves precede the harmonic comb, uncertainty
floor and LPC weighting. LPC compares broad spectral shapes with enrollment exemplars and applies
a confidence-weighted multiplier before the mask ratio. In this replay, {branch}. The 2 ms
cepstral lifter, when active, adds an envelope-based estimate to the male mask. Cepstral pitch
evidence remains active with either profile type. The overlap score uses the shared bins where
both unexponentiated synthesis combs are at least 0.5. It describes coincident harmonics, and the
two positive estimates alone cannot establish simultaneous speech. Shared harmonics and mixture
phase limit separation.

**Likely viva questions**

1. Why sum squared magnitudes? The fit estimates how spectral power distributes across bands.
2. How does LPC influence the mask? Spectral shapes from enrollment scale the provisional speaker strengths before normalization.
3. Why can the masks sum to a value other than one? The optional cepstral correction changes the male mask independently of the female mask.

## Figure 3 · Comparative spectrograms and STE

**Syllabus.** Time–frequency analysis, short-time energy and signal attenuation.
STE[t] = sum(x[t,n]^2)/640. Total attenuation = 10 log10(sum(x_mix^2)/sum(x_selected^2)).
A periodic Hann and 1024-point FFT with 10 ms hops are used for all spectrograms. Color shows
Hann coherent-gain-corrected, single-sided amplitude in dBFS with the same {FLOOR} to 0 dBFS limits.

**Interpretation.** {amp_note} Compare the spectra and energy traces by authorized interval.
Dashed lines are saved sample-index events and shaded intervals follow the saved routes. Automatic
switches appear among those events, though the current log does not distinguish timer events
from manual Next actions. This total energy difference includes speech, noise, overlap, and any
muted periods. True interfering-speaker attenuation requires an isolated reference for that speaker.

**Likely viva questions**

1. Why fix the color scale? A quieter recording should look quieter at the same absolute reference.
2. Why can STE and spectral colors tell different stories? STE summarizes the whole frame, while the spectrogram resolves energy by time and frequency.
3. Can the energy ratio measure rejection of the other speaker? It combines every contribution and cannot identify source-specific rejection.
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="folder containing the four WAVs and turns.csv")
    parser.add_argument("--profiles", required=True, type=Path, help="profile JSON used for the session")
    parser.add_argument("--male-calibration", type=Path, help="isolated male enrollment WAV")
    parser.add_argument("--female-calibration", type=Path, help="isolated female enrollment WAV")
    args = parser.parse_args()
    try:
        folder = analyze(args.session, args.profiles, {
            "male": args.male_calibration, "female": args.female_calibration})
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")
    print(folder.resolve())


if __name__ == "__main__":
    main()

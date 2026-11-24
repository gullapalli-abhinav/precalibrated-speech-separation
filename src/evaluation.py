"""Repeatable held-out mixtures, aligned metrics and reference-free diagnostics."""

import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from config import METHODS, SPEAKERS, Settings
from src.audio_io import partition, read_audio, rms, write_audio
from src.calibration import load_profiles, write_json
from src.pitch_tracking import frames, single_pitch
from src.separator import Analysis, Separation, analyze, save_outputs, separate

RATIOS = (-10, -5, 0, 5, 10)
MIX_FILENAME = "Digi_added_samples.wav"
PCM_SCALE = 2 ** 31
HEADROOM = 10 ** (-1 / 20)


def controlled_wavs(directory: Path, sources: np.ndarray, ratio: int | None, cfg: Settings) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep reference quantization exact by summing PCM32 integers."""
    source_rms = np.sqrt(np.mean(sources ** 2, axis=1))
    target_rms = np.full(2, 0.10)
    if ratio is not None:
        if np.any(source_rms < 1e-6):
            raise ValueError(f"{directory.name}: a held-out speaker has insufficient level")
        target_rms *= 10 ** (np.array([ratio, -ratio]) / 40)
    gains = np.divide(target_rms, source_rms, out=np.zeros(2), where=source_rms >= 1e-6)
    gained = sources * gains[:, None]
    peak = max(np.max(np.abs(gained)), np.max(np.abs(gained.sum(axis=0))))
    if peak == 0:
        raise ValueError(f"{directory.name}: no usable held-out audio")
    common = min(1.0, HEADROOM / peak)
    pcm = np.rint(gained * common * PCM_SCALE).astype(np.int32)
    mixed = pcm.astype(np.int64).sum(axis=0)
    if np.max(np.abs(mixed)) >= PCM_SCALE:
        raise ValueError(f"{directory.name}: mixture would exceed PCM32 range")
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"reference_{speaker}.wav" for speaker in SPEAKERS]
    for path, reference in zip(paths, pcm):
        wavfile.write(path, cfg.rate, reference)
    mixture_file = directory / MIX_FILENAME
    wavfile.write(mixture_file, cfg.rate, mixed.astype(np.int32))
    x, _ = read_audio(mixture_file, cfg.rate)
    references = np.stack([read_audio(path, cfg.rate)[0] for path in paths])
    if not np.array_equal(x, references.sum(axis=0)):
        raise ValueError(f"{directory.name}: on-disk mixture does not equal the saved reference sum")
    saved_rms = [rms(s) for s in references]
    record = {
        "mixture": MIX_FILENAME,
        "references": [path.name for path in paths],
        "sample_rate_hz": cfg.rate,
        "samples": len(x),
        "format": "PCM32 WAV",
        "normalization_gains": gains.tolist(),
        "common_headroom_gain": common,
        "reference_rounding_max_abs": 0.5 / PCM_SCALE,
        "mixture_sum_max_abs_error": float(np.max(np.abs(x - references.sum(axis=0)))),
        "mixture_peak": float(np.max(np.abs(x))),
        "reference_peak": np.max(np.abs(references), axis=1).tolist(),
        "reference_rms": saved_rms,
        "requested_female_to_male_db": ratio,
        "measured_female_to_male_db": float(20 * np.log10(saved_rms[0] / saved_rms[1])) if min(saved_rms) > 0 else None,
    }
    return x, references, record


def power_db(numerator: float, denominator: float) -> float | None:
    if numerator == denominator == 0:
        return None
    if numerator == 0:
        return -120.0
    if denominator == 0:
        return 120.0
    return float(np.clip(10 * np.log10(numerator / denominator), -120, 120))


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float | None:
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    target_energy = float(target @ target)
    if target_energy == 0 or not np.any(estimate):
        return None
    projection = (estimate @ target / target_energy) * target
    error = estimate - projection
    return power_db(projection @ projection, error @ error)


def projection_gains(estimate: np.ndarray, references: np.ndarray) -> list[float | None]:
    """Joint regression measures source-correlated amplitude independently of residual."""
    estimate = estimate - estimate.mean()
    refs = references - references.mean(axis=1, keepdims=True)
    energy = np.linalg.norm(refs, axis=1)
    valid = energy > 1e-12
    gains = [None, None]
    if valid.any():
        basis = refs[valid] / energy[valid, None]
        gram = basis @ basis.T
        if np.linalg.cond(gram) < 1e6:
            coefficients = np.linalg.solve(gram, basis @ estimate) / energy[valid]
            for index, gain in zip(np.flatnonzero(valid), coefficients):
                gains[index] = power_db(gain ** 2, 1.0)
    return gains


def score(estimate: np.ndarray, x: np.ndarray, references: np.ndarray, target: int) -> dict:
    baseline = si_sdr(x, references[target])
    quality = si_sdr(estimate, references[target])
    gains = projection_gains(estimate, references)
    return {
        "si_sdr_db": quality,
        "mixture_si_sdr_db": baseline,
        "si_sdri_db": quality - baseline if quality is not None and baseline is not None and baseline < 120 else None,
        "target_gain_db": gains[target],
        "interferer_attenuation_db": -gains[1 - target] if gains[1 - target] is not None else None,
        "output_to_input_rms_db": power_db(estimate @ estimate, x @ x),
    }


def reference_conditions(references: np.ndarray, cfg: Settings) -> tuple[dict, np.ndarray]:
    data = [frames(ref, cfg) for ref in references]
    f0 = np.column_stack([single_pitch(d, cfg)[0] for d in data])
    voiced = f0 > 0
    active = np.column_stack([d.active for d in data])
    noisy = active & ~voiced & (np.column_stack([d.zcr for d in data]) >= cfg.unvoiced_zcr)
    ratio = np.max(f0, axis=1) / np.maximum(1, np.min(f0, axis=1))
    close = ratio <= 2 ** (2 / 12)
    related = np.min(np.abs(ratio[:, None] / np.array([1.5, 2, 3]) - 1), axis=1) <= 0.04
    overlap = active.all(axis=1)
    both_voiced = voiced.all(axis=1)
    return {
        "both_active": overlap,
        "female_active_only": active[:, 0] & ~active[:, 1],
        "male_active_only": active[:, 1] & ~active[:, 0],
        "both_voiced": both_voiced,
        "close_f0": both_voiced & close,
        "harmonically_related_f0": both_voiced & related,
        "distinct_f0": both_voiced & ~close & ~related,
        "high_zcr_unvoiced_overlap": overlap & noisy.any(axis=1),
    }, f0


def pitch_comparison(hypothesis: np.ndarray, reference: np.ndarray) -> dict:
    output = {}
    for i, speaker in enumerate(SPEAKERS):
        valid = reference[:, i] > 0
        found = valid & (hypothesis[:, i] > 0)
        errors = np.abs(hypothesis[found, i] / reference[found, i] - 1)
        output[speaker] = {
            "reference_voiced_frames": int(valid.sum()),
            "both_report_f0_frames": int(found.sum()),
            "within_10pct_of_reference_fraction": float((errors <= 0.10).sum() / valid.sum()) if valid.any() else None,
            "median_relative_error_when_both_report_f0": float(np.median(errors)) if len(errors) else None,
        }
    return output


def summary(x: np.ndarray, analysis: Analysis, result: Separation, cfg: Settings) -> dict:
    active = analysis.frames.active
    duration = len(x) / cfg.rate
    seconds = analysis.seconds + result.seconds
    return {
        "duration_seconds": duration,
        "analysis_frames": len(active),
        "activity_frames": int(active.sum()),
        "likely_unvoiced_active_frames": int(np.count_nonzero(
            active & ~(analysis.tracks.f0 > 0).any(axis=1) & (analysis.frames.zcr >= cfg.unvoiced_zcr)
        )),
        "analysis_seconds": analysis.seconds,
        "synthesis_seconds": result.seconds,
        "processing_seconds": seconds,
        "milliseconds_per_frame": 1000 * seconds / len(active),
        "real_time_factor": seconds / duration,
        "mixture_reconstruction_max_abs_error": float(np.max(np.abs(result.audio.sum(axis=0) - x))),
        "dual_pitch_fraction_active": float(np.mean((analysis.tracks.f0[active] > 0).all(axis=1))) if active.any() else None,
        "speakers": {
            name: {
                "pitch_frames": int(np.count_nonzero(analysis.tracks.f0[:, i])),
                "median_pitch_hz": float(np.median(analysis.tracks.f0[analysis.tracks.f0[:, i] > 0, i]))
                if np.any(analysis.tracks.f0[:, i]) else None,
                "output_rms": rms(result.audio[i]),
                "output_peak": float(np.max(np.abs(result.audio[i]))),
            }
            for i, name in enumerate(SPEAKERS)
        },
    }


def save_tracks(path: Path, analysis: Analysis, cfg: Settings) -> None:
    f0 = analysis.tracks.f0
    voiced = f0 > 0
    status = np.where(voiced, "voiced", np.where(
        ~analysis.frames.active[:, None], "silent",
        np.where(analysis.frames.zcr[:, None] >= cfg.unvoiced_zcr, "likely_unvoiced", "uncertain"),
    ))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "frame_energy", "zcr", "female_f0_hz", "male_f0_hz",
                         "female_confidence", "male_confidence", "female_status", "male_status"])
        for i, (pitches, confidence, labels) in enumerate(zip(f0, analysis.tracks.confidence, status)):
            writer.writerow([round(i * cfg.hop / cfg.rate, 6), analysis.frames.energy[i], analysis.frames.zcr[i],
                             *pitches, *confidence, *labels])


def evaluate(
    sources: tuple[Path, Path], real: Path, legacy: Path | None, profile_path: Path, directory: Path
) -> dict:
    from src.visualization import comparison_plot, separation_plots

    digest = sha256(profile_path.read_bytes()).hexdigest() if profile_path.is_file() else None
    profiles, cfg = load_profiles(profile_path)
    held = []
    ranges = {}
    for name, path in zip(SPEAKERS, sources):
        _, test, info = partition(path, cfg)
        previous = profiles[name]["recording"]
        keys = ("sha256", "processing_rate_hz", "calibration_samples", "held_out_samples")
        if any(info[key] != previous[key] for key in keys):
            raise ValueError(f"{path.name}: held-out source or split does not match the fixed {name} profile")
        held.append(test)
        ranges[name] = {key: info[key] for key in (*keys, "rate_hz", "source_calibration_samples", "source_held_out_samples")}
    n = min(map(len, held))
    aligned = np.stack([x[:n] for x in held])
    offset = 2 * cfg.rate
    staggered = np.stack([np.pad(aligned[0], (0, offset)), np.pad(aligned[1], (offset, 0))])
    cases = {f"ratio_{ratio:+d}dB": (aligned, ratio, [0, 0]) for ratio in RATIOS}
    cases.update({
        "female_only": (np.stack([aligned[0], np.zeros(n)]), None, [0, None]),
        "male_only": (np.stack([np.zeros(n), aligned[1]]), None, [None, 0]),
        "staggered_0dB": (staggered, 0, [0, offset]),
    })
    report = {"profile_sha256": digest, "partitions": ranges, "controlled": {}, "reference_free": {}}
    rows = []
    for label, (sources_for_case, ratio, offsets) in cases.items():
        target = directory / "controlled" / label
        x, refs, manifest = controlled_wavs(target, sources_for_case, ratio, cfg)
        manifest["used_held_out_ranges"] = {
            name: {
                "start_sample": ranges[name]["held_out_samples"][0],
                "end_sample_exclusive": ranges[name]["held_out_samples"][0] + n,
                "offset_in_mixture_samples": pos,
            } if pos is not None else None
            for name, pos in zip(SPEAKERS, offsets)
        }
        write_json(target / "mixture.json", manifest)
        analysis = analyze(x, profiles, cfg)
        save_tracks(target / "pitch.csv", analysis, cfg)
        conditions, reference_f0 = reference_conditions(refs, cfg)
        # Assign each sample to the nearest centered analysis frame.
        indices = (np.arange(len(x)) + cfg.hop // 2) // cfg.hop
        subsets = {key: value[indices] for key, value in conditions.items()}
        subset_seconds = {key: float(np.count_nonzero(value) / cfg.rate) for key, value in subsets.items()}
        runs = {}
        for method in METHODS:
            result = separate(analysis, len(x), profiles, cfg, method)
            save_outputs(target / method, x, result, cfg, "both")
            runs[method] = summary(x, analysis, result, cfg)
            # Score the actual float32 estimate as saved.
            estimates = result.audio.astype(np.float32).astype(np.float64)
            for subgroup, mask in {"all": slice(None), **subsets}.items():
                if subgroup != "all" and subset_seconds[subgroup] < 0.25:
                    continue
                for i, name in enumerate(SPEAKERS):
                    metrics = score(estimates[i, mask], x[mask], refs[:, mask], i)
                    rows.append({"condition": label, "subset": subgroup, "method": method, "target": name,
                                 "seconds": len(x[mask]) / cfg.rate, **metrics})
            if label == "ratio_+0dB":
                separation_plots(target / method, x, analysis, result, cfg, f"Controlled 0 dB, {method}")
        report["controlled"][label] = {
            **manifest, "subgroup_seconds": subset_seconds,
            "pitch_against_isolated_estimates": pitch_comparison(analysis.tracks.f0, reference_f0),
            "methods": runs,
        }
        print(f"Evaluated {label} ({len(x) / cfg.rate:.2f} s)", flush=True)
    for label, path in (("real", real), ("supplied_digital", legacy)):
        if path is None:
            continue
        x, info = read_audio(path, cfg.rate)
        analysis = analyze(x, profiles, cfg)
        target = directory / label
        target.mkdir(parents=True, exist_ok=True)
        save_tracks(target / "pitch.csv", analysis, cfg)
        if label == "real":
            write_audio(target / "output_silence.wav", np.zeros_like(x), cfg.rate)
        runs = {}
        for method in METHODS:
            result = separate(analysis, len(x), profiles, cfg, method)
            save_outputs(target / method, x, result, cfg, "both")
            runs[method] = summary(x, analysis, result, cfg)
            if label == "real":
                separation_plots(target / method, x, analysis, result, cfg, f"{path.name}, {method}")
        report["reference_free"][label] = {"recording": info, "methods": runs}
        print(f"Processed {label} ({len(x) / cfg.rate:.2f} s)", flush=True)
    if sha256(profile_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Calibration profiles changed during evaluation")
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["metrics"] = rows
    write_json(directory / "evaluation.json", report)
    comparison_plot(directory / "method_comparison.png", rows)
    return report

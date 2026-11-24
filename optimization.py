"""Offline controlled experiments. Isolated references are only used in this evaluator."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy import fft

from config import DEFAULT, FEMALE, MALE, OUTPUT, REAL, SPEAKERS
from src.audio_io import partition, read_audio
from src.calibration import create_profiles, load_profiles, write_json
from src.evaluation import RATIOS, controlled_wavs, reference_conditions, score
from src.pitch_tracking import Tracks, frames, harmonic_grid, periodicity, track
from src.separator import Analysis, analyze, lpc_evidence, power_strengths, save_outputs, separate, synthesize

DEST = OUTPUT / "optimization"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profiles_and_audio(split):
    # Development ends before the 60% boundary used by historical calibration.
    cfg = replace(DEFAULT, calibration_fraction=0.4) if split == "development" else DEFAULT
    folder = DEST / split
    profile_path = folder / "profiles.json"
    profiles, _ = create_profiles((FEMALE, MALE), profile_path, cfg)
    held = [partition(p, cfg)[1] for p in (FEMALE, MALE)]
    n = 12 * cfg.rate if split == "development" else min(map(len, held))
    for name in SPEAKERS:
        rec = profiles[name]["recording"]
        if split == "development":
            assert rec["held_out_samples"][0] + n < int(0.60 * rec["source_held_out_samples"][1])
    start, stop = 0, n
    if split == "validation":
        stop = 6 * cfg.rate
    elif split == "test":
        start = int(6.25 * cfg.rate)
    write_json(folder / "partition.json", {
        name: [profiles[name]["recording"]["held_out_samples"][0] + offset for offset in (start, stop)]
        for name in SPEAKERS})
    return profiles, cfg, np.stack([h[start:stop] for h in held])


def conditions(aligned, cfg):
    for ratio in RATIOS:
        yield f"ratio_{ratio:+d}dB", ratio, aligned
    offset = 2 * cfg.rate
    yield "staggered_0dB", 0, np.stack([np.pad(aligned[0], (0, offset)), np.pad(aligned[1], (offset, 0))])
    for i, name in enumerate(SPEAKERS):
        single = aligned.copy()
        single[1-i] = 0
        yield name + "_only", None, single


def pitch_stats(tracks, ref, active, labels, diag):
    hyp = tracks.f0
    both = labels["both_voiced"]
    found = hyp > 0
    valid = ref > 0
    ratio = hyp / np.maximum(ref, 1)
    correct = np.abs(ratio - 1) <= 0.10
    direct = correct.all(1)
    cross = (np.abs(hyp / np.maximum(ref[:, ::-1], 1) - 1) <= 0.10).all(1)
    # Exclude ambiguous direct/cross matches and reset continuity across gaps.
    assigned = both & (direct ^ cross)
    switched = assigned[1:] & assigned[:-1] & (cross[1:] != cross[:-1])
    def fraction(v, where):
        return float(v[where].mean()) if where.any() else None
    summary = dict(frames=len(ref), both_active_fraction=labels["both_active"].mean(),
                   both_voiced_frames=int(both.sum()), both_voiced_fraction=both.mean(),
                   one_voiced_fraction=(valid.sum(1) == 1).mean(), no_voiced_fraction=(~valid.any(1)).mean(),
                   dual_on_active=fraction(found.all(1), active), dual_on_both=fraction(found.all(1), both),
                   correct_pair_on_both=fraction(direct, both), swapped_frames=int((assigned & cross).sum()),
                   identity_switches=int(switched.sum()))
    if diag:
        available = (np.abs(diag["choices"] / np.maximum(ref[:, :, None], 1) - 1) <= 0.1).any(-1)
        summary.update(candidate_pair_on_both=fraction(available.all(1), both),
                       greedy_dual_on_both=fraction((diag["frame_best"] > 0).all(1), both))
    for i, name in enumerate(SPEAKERS):
        output = dict(accuracy=fraction(correct[:, i], valid[:, i]),
                      accuracy_on_both=fraction(correct[:, i], both),
                      missed=int((valid[:, i] & ~found[:, i]).sum()),
                      spurious=int((~valid[:, i] & found[:, i]).sum()),
                      octave_low=int((valid[:, i] & (np.abs(ratio[:, i] / 0.5 - 1) <= 0.1)).sum()),
                      octave_high=int((valid[:, i] & (np.abs(ratio[:, i] / 2 - 1) <= 0.1)).sum()))
        if diag:
            output["candidate_on_both"] = fraction(available[:, i], both)
        summary.update({name + "_" + k: v for k, v in output.items()})
    return summary


def track_trials():
    p, c, sources = profiles_and_audio("development")
    rows = []
    for condition, ratio, signals in conditions(sources, c):
        x, refs, _ = controlled_wavs(DEST / "development" / condition, signals, ratio, c)
        d = frames(x, c)
        labels, ref = reference_conditions(refs, c)
        for name, legacy, transition in [("original", True, 1), ("reduced_transition", True, .35),
                                          ("scaled_evidence", False, 1), ("combined", False, .35),
                                          ("refined_peaks", False, .35)]:
            diag = {}
            t = track(d, p, c, legacy=legacy, transition_scale=transition, refined=name=="refined_peaks", diagnostics=diag)
            rows.append(dict(condition=condition, variant=name, **pitch_stats(t, ref, d.active, labels, diag)))
    write_rows(DEST / "pitch_diagnostics" / "development_tracker.csv", rows)


def evaluate_split(split):
    profiles, cfg, aligned = profiles_and_audio(split)
    metrics, pitches = [], []
    freq = fft.rfftfreq(cfg.fft, 1/cfg.rate)
    for condition, ratio, signals in conditions(aligned, cfg):
        folder = DEST / split / condition
        x, refs, manifest = controlled_wavs(folder, signals, ratio, cfg)
        write_json(folder / "mixture.json", manifest)
        labels, ref = reference_conditions(refs, cfg)
        start = perf_counter()
        data = frames(x, cfg)
        frame_time = perf_counter() - start
        for tracker in ("original", "improved", "oracle"):
            start = perf_counter()
            diag = {}
            if tracker == "oracle":
                tracks = Tracks(ref, (ref > 0).astype(float))
            else:
                # The mask development study preceded the peak-resolution trial.
                tracks = track(data, profiles, cfg, legacy=tracker=="original",
                               refined=tracker=="improved" and split!="development", diagnostics=diag)
            likelihood = harmonic_grid(tracks.f0, freq, cfg.rate/cfg.fft)
            a = Analysis(data, tracks, likelihood, perf_counter() - start + frame_time)
            if tracker != "oracle":
                pitches.append(dict(condition=condition, tracker=tracker,
                                    **pitch_stats(tracks, ref, data.active, labels, diag)))
            variants = ["legacy", "conservative", "power", "full_lpc"]
            if tracker == "improved":
                variants += ["lpc_only", "half_lpc"]
            for variant in variants:
                start = perf_counter()
                if variant == "full_lpc":
                    result = separate(a, len(x), profiles, cfg, "harmonic_lpc")
                elif variant in ("half_lpc", "lpc_only"):
                    if variant == "lpc_only":
                        neutral = replace(a, tracks=Tracks(tracks.f0, np.ones_like(tracks.confidence)))
                        strengths = lpc_evidence(neutral, profiles, cfg)
                    else:
                        strengths = power_strengths(a, profiles, cfg, True)
                        strengths *= np.sqrt(lpc_evidence(a, profiles, cfg))
                    result = synthesize(a, strengths, len(x), cfg, start)
                else:
                    result = separate(a, len(x), profiles, cfg, "harmonic", mask=variant)
                assert result.audio.shape == refs.shape and np.isfinite(result.audio).all()
                error = float(np.max(np.abs(result.audio.sum(0) - x)))
                assert error < 1e-12
                seconds = a.seconds + result.seconds
                for i, target in enumerate(SPEAKERS):
                    saved = result.audio[i].astype(np.float32).astype(float)
                    metrics.append(dict(split=split, condition=condition, tracker=tracker, mask=variant,
                                        target=target, **score(saved, x, refs, i), seconds=seconds,
                                        rtf=seconds/(len(x)/cfg.rate), reconstruction_error=error))
                if split in ("test", "held_out") and tracker == "improved" and variant == "full_lpc":
                    save_outputs(folder, x, result, cfg, "both")
        print(f"Scored {split} {condition}", flush=True)
    write_rows(DEST / "mask_comparison" / (split + ".csv"), metrics)
    write_rows(DEST / "pitch_diagnostics" / (split + ".csv"), pitches)
    if split in ("test", "held_out"):
        collect_results()


def collect_results():
    ablations, pitch_rows = [], []
    for split in ("test", "held_out"):
        path = DEST / "mask_comparison" / (split + ".csv")
        if path.exists():
            with path.open() as file:
                ablations.extend(csv.DictReader(file))
            with (DEST / "pitch_diagnostics" / (split + ".csv")).open() as file:
                pitch_rows.extend(dict(r, split=split) for r in csv.DictReader(file))
    write_rows(DEST / "ablation_results.csv", ablations)
    for t, m, filename in [("original", "legacy", "baseline_metrics"), ("improved", "full_lpc", "optimized_metrics")]:
        selected = [r for r in ablations if r["tracker"] == t and r["mask"] == m]
        diagnostics = {(r["split"], r["condition"]): r for r in pitch_rows if r["tracker"] == t}
        report = []
        for row in selected:
            cohort = [r for r in selected if r["split"] == row["split"]]
            pair = [float(r["si_sdri_db"]) for r in cohort
                    if r["condition"] == row["condition"] and r["si_sdri_db"]]
            sweep = [float(r["si_sdri_db"]) for r in cohort if r["condition"].startswith("ratio_")]
            report.append(row | diagnostics[row["split"], row["condition"]] | dict(
                pair_mean_si_sdri_db=np.mean(pair) if pair else None,
                five_ratio_mean_si_sdri_db=np.mean(sweep), five_ratio_worst_target_si_sdri_db=min(sweep),
                target_attenuation_db=-float(row["target_gain_db"]) if row["target_gain_db"] else None,
                interferer_leakage_gain_db=-float(row["interferer_attenuation_db"]) if row["interferer_attenuation_db"] else None))
        write_rows(DEST / (filename + ".csv"), report)


def verify():
    """Cross-check independent CLI renders and the scored PCM on disk."""
    def digest(path):
        return sha256(path.read_bytes()).hexdigest()
    def load(path):
        return read_audio(path)[0]
    def read_rows(path):
        with path.open() as file:
            return list(csv.DictReader(file))
    def index(row):
        return row["condition"], row["method"], row["subset"], row["target"]

    with ZipFile(DEST / "baseline_snapshot.zip") as archive:
        history = {index(r): r for r in csv.DictReader(archive.read("metrics.csv").decode().splitlines())}
    reproduction = {index(r): r for r in read_rows(DEST / "reproduction" / "metrics.csv")}
    assert history.keys() == reproduction.keys()
    deviation = max(abs(float(old[k]) - float(reproduction[key][k])) for key, old in history.items()
                    for k in old if k.endswith("_db") and old[k])
    assert deviation < 1e-7
    final_profile = DEST / "final_cli" / "profiles.json"
    profiles, cfg = load_profiles(final_profile)
    for speaker, source in zip(SPEAKERS, (FEMALE, MALE)):
        assert digest(source) == profiles[speaker]["recording"]["sha256"]
    audit = {"historical_rows_verified": len(history), "largest_historical_metric_delta_db": deviation,
             "final_profile_sha256": digest(final_profile), "controlled": {}}
    for folder in sorted((DEST / "reproduction" / "controlled").iterdir()):
        final = DEST / "held_out" / folder.name
        cli = DEST / "final_cli" / "controlled" / folder.name / "harmonic_lpc"
        for filename in ("Digi_added_samples.wav", "reference_female.wav", "reference_male.wav"):
            assert digest(folder/filename) == digest(final/filename)
        x = load(final/"Digi_added_samples.wav")
        outputs = []
        for name in SPEAKERS:
            filename = f"output_{name}.wav"
            y = load(final/filename)
            assert np.array_equal(y, load(cli/filename))
            outputs.append(y)
        residual = float(np.max(np.abs(np.sum(outputs, axis=0) - x)))
        assert residual < 1e-7
        audit["controlled"][folder.name] = dict(samples=len(x), saved_sum_error=residual)
    saved = DEST / "test" / "cli_mute_male"
    assert np.array_equal(load(saved/"output_selected.wav"), load(DEST/"test"/"ratio_+0dB"/"output_female.wav"))
    audit["strict_test_rerender_and_mute_male_exact"] = True
    validation, test = [json.loads((DEST / part / "partition.json").read_text()) for part in ("validation", "test")]
    for s in SPEAKERS:
        assert test[s][0]-validation[s][1] == round(.25*cfg.rate)
    balanced = DEST / "test" / "ratio_+0dB"
    x = load(balanced / "Digi_added_samples.wav")
    data = frames(x, cfg)
    centers = np.arange(len(data.energy)) * cfg.hop
    valid = centers < len(x)
    assert np.array_equal(data.samples[valid, cfg.window//2], x[centers[valid]])
    ac, _ = periodicity(data, cfg)
    worst = 0.
    for i in np.linspace(1, len(centers)-2, 20, dtype=int):
        frame = data.samples[i] - data.samples[i].mean()
        for lag in range(int(cfg.rate/cfg.pitch_max), int(cfg.rate/cfg.pitch_min)+1):
            left, right = frame[:-lag], frame[lag:]
            expected = (left @ right) / max(np.linalg.norm(left)*np.linalg.norm(right), 1e-12)
            worst = max(worst, abs(expected-ac[i, lag]))
    assert worst < 1e-10
    audit["max_normalized_ac_difference_direct_slices"] = float(worst)
    audit["valid_test_centers_aligned"] = int(valid.sum())
    write_json(DEST/"verification.json", audit)


def benchmark():
    profiles, cfg = load_profiles(DEST/"final_cli"/"profiles.json")
    x = read_audio(REAL)[0]
    rows = []
    for run in range(3):
        for mode in (("original", "improved") if run != 1 else ("improved", "original")):
            start = perf_counter()
            if mode == "original":
                data = frames(x, cfg)
                tracks = track(data, profiles, cfg, legacy=True)
                likelihood = harmonic_grid(tracks.f0, fft.rfftfreq(cfg.fft, 1/cfg.rate), cfg.rate/cfg.fft)
                analysis = Analysis(data, tracks, likelihood, 0)
                result = separate(analysis, len(x), profiles, cfg, "harmonic", mask="legacy")
            else:
                analysis = analyze(x, profiles, cfg)
                result = separate(analysis, len(x), profiles, cfg, "harmonic_lpc")
            elapsed = perf_counter()-start
            rows.append(dict(run=run+1, method=mode, seconds=elapsed, rtf=elapsed/(len(x)/cfg.rate)))
            del analysis, result
    write_rows(DEST/"real_timing.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("tracker", "development", "validation", "test", "held_out", "verify", "benchmark"))
    study = parser.parse_args().study
    if study == "tracker":
        track_trials()
    elif study == "verify":
        verify()
    elif study == "benchmark":
        benchmark()
    else:
        evaluate_split(study)

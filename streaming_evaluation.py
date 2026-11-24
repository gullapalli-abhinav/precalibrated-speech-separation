"""Incremental replay with the original disjoint source partitions and aligned WAV references."""
import argparse
import itertools
from time import perf_counter, process_time

import numpy as np
import psutil
from scipy import signal

from config import OUTPUT, SPEAKERS
from male_study import BANDS, ROOT, calibrated_profiles, envelope_difference, read_case
from optimization import DEST, write_rows
from src.audio_io import read_audio
from src.calibration import load_profiles
from src.cepstrum import log_envelope
from src.evaluation import RATIOS, projection_gains, score
from src.pitch_tracking import frames
from src.separator import analyze, separate
from src.streaming import StreamingSeparator

DIRECTORY = OUTPUT / "realtime"


def replay(engine, x, sizes=(1, 127, 320, 79, 512, 160)):
    pos = 0
    blocks = []
    for size in itertools.cycle(sizes):
        if pos >= len(x):
            break
        blocks.append(engine.feed(x[pos:pos+size]))
        pos += size
    blocks.append(engine.finish())
    return np.concatenate(blocks).T


def single_speaker_cases(split, profiles, cfg):
    singles = []
    for i,s in enumerate(SPEAKERS):
        source = DEST/split/f"{s}_only"/"Digi_added_samples.wav"
        x,_ = read_audio(source)
        y = replay(StreamingSeparator(profiles,cfg),x).astype(float)
        singles.append(dict(source=s, target_output_db=10*np.log10(np.sum(y[i]**2)/np.sum(x**2)),
            unauthorized_output_db=10*np.log10(np.sum(y[1-i]**2)/np.sum(x**2))))
    write_rows(DIRECTORY/"single_speaker.csv",singles)


def evaluate(split, final):
    _, cfg = load_profiles(DEST/split/"profiles.json")
    path = ROOT/"comparisons"/"profiles"/f"{split}_{cfg.window}.json"
    if not path.exists():
        calibrated_profiles(split, cfg, cfg.window)
    profiles, cfg = load_profiles(path)
    rows, timings = [], []
    for ratio in (RATIOS if final else [0]):
        x, refs = read_case(split, ratio)
        male = frames(refs[1], cfg)
        reference_log = log_envelope(male.spectrum, cfg, 2.)
        analysis = analyze(x, profiles, cfg)
        outputs = {}
        for variant in ("current", "cepstral_assisted"):
            outputs["offline_"+variant] = separate(analysis, len(x), profiles, cfg, "harmonic_lpc",
                                                   male_variant=variant).audio
        trials = [(2, "bounded")] if final else [(2, "causal"), (2, "bounded"), (3, "bounded"), (5, "bounded")]
        for lag, smoothing in trials:
            name = f"lag{lag}_{smoothing}"
            engine = StreamingSeparator(profiles, cfg, lag, smoothing)
            process = psutil.Process()
            start, cpu = perf_counter(), process_time()
            outputs[name] = replay(engine, x).astype(float)
            wall = perf_counter()-start
            times = np.array(engine.times)*1000
            timings.append(dict(split=split, ratio=ratio, mode=name, frames=engine.frame_count,
                median_ms=np.median(times), p95_ms=np.percentile(times,95), p99_ms=np.percentile(times,99),
                max_ms=times.max(), misses=engine.misses, rtf=wall/(len(x)/cfg.rate),
                cpu_core_percent=100*(process_time()-cpu)/wall, rss_mib=process.memory_info().rss/2**20,
                delay_ms=1000*engine.delay/cfg.rate))
        for name, y in outputs.items():
            for i, speaker in enumerate(SPEAKERS):
                row = dict(split=split, ratio=ratio, mode=name, speaker=speaker,
                           **score(y[i], x, refs, i))
                row["male_envelope_rmse_db"] = (envelope_difference(frames(y[i],cfg).spectrum,
                    reference_log, male.active, cfg) if i == 1 else None)
                for low, high in BANDS:
                    if high == cfg.rate/2:
                        sos = signal.butter(4, low, fs=cfg.rate, btype="highpass", output="sos")
                    else:
                        sos = signal.butter(4, (low, high), fs=cfg.rate, btype="bandpass", output="sos")
                    row[f"target_gain_{low}_{high}_db"] = projection_gains(
                        signal.sosfilt(sos, y[i]), signal.sosfilt(sos, refs))[i]
                rows.append(row)
        print(split, ratio, flush=True)
    if split == "development":
        single_speaker_cases(split, profiles, cfg)
    write_rows(DIRECTORY/f"{split}_quality.csv", rows)
    write_rows(DIRECTORY/f"{split}_timing.csv", timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "final", "debate"])
    stage = parser.parse_args().stage
    if stage == "debate":
        from debate_diagnostics import render
        render(DIRECTORY, replay)
    else:
        for split in (["test"] if stage == "final" else ["development", "validation"]):
            evaluate(split, stage == "final")

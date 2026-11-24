"""Prepare time-aligned raw and loudness-matched mono WAV comparisons."""

import numpy as np
from scipy import signal

from optimization import write_rows
from src.audio_io import read_audio, write_audio


def loudness(audio, rate):
    """Mono integrated loudness using K weighting and BS.1770 block gates."""
    # Use the published 48 kHz biquads after polyphase resampling.
    gcd = np.gcd(48000, rate)
    x = signal.resample_poly(audio, 48000//gcd, rate//gcd)
    shelf = [1.53512485958697, -2.69169618940638, 1.19839281085285,
             1., -1.69065929318241, .73248077421585]
    highpass = [1., -2., 1., 1., -1.99004745483398, .99007225036621]
    x = signal.sosfilt([shelf, highpass], x)
    if len(x) < 19200:
        x = np.pad(x, (0, 19200-len(x)))
    power = np.lib.stride_tricks.sliding_window_view(x*x, 19200)[::4800].mean(axis=1)
    audible = power[power > 10**((-70+.691)/10)]
    if not audible.size:
        raise ValueError("A silent comparison cannot be loudness matched")
    gated = audible[audible >= audible.mean()*.1]
    return float(-.691 + 10*np.log10(gated.mean()))


def listening_set(folder, signals, rate):
    """Write unity-gain masters first, then match the group at common headroom."""
    levels = {name: loudness(y, rate) for name,y in signals.items()}
    peaks = {name: float(np.max(abs(signal.resample_poly(y,4,1)))) for name,y in signals.items()}
    # A shared target respects 1 dB of oversampled peak headroom on every file.
    target = min(-23., *(levels[name] - 1 - 20*np.log10(peak) for name,peak in peaks.items()))
    rows, sheet = [], []
    n = len(next(iter(signals.values())))
    for name, raw in signals.items():
        assert len(raw) == n
        write_audio(folder/"raw"/(name+".wav"), raw, rate)
        gain_db = target-levels[name]
        for _ in range(3):
            audio = raw * 10**(gain_db/20)
            measured = loudness(audio, rate)
            if abs(target-measured) < .015:
                break
            gain_db += target-measured
        path = folder/(name+".wav")
        write_audio(path,audio,rate)
        saved = read_audio(path,rate)[0]
        peak = np.max(abs(signal.resample_poly(saved,4,1)))
        level = loudness(saved,rate)
        assert len(saved)==n and peak < 1 and abs(level-target) < .08
        rows.append(dict(file=path.name,samples=n,rate=rate,raw_lufs=levels[name],gain_db=gain_db,
                         target_lufs=target,matched_lufs=level,oversampled_peak=peak))
        if name.startswith("male_"):
            sheet.append({"Variant name":name, "Male intelligibility (1-5)":"",
                          "Male naturalness (1-5)":"", "Female leakage (1-5)":"", "Notes":""})
    write_rows(folder/"levels.csv",rows)
    write_rows(folder/"listening_sheet.csv",sheet)

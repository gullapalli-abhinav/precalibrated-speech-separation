"""Offline figures and allocation checks on the balanced, aligned mixture."""

import argparse
from dataclasses import replace
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from config import SPEAKERS
from optimization import DEST, write_rows
from src.audio_io import read_audio
from src.calibration import load_profiles, write_json
from src.evaluation import RATIOS, reference_conditions, score
from src.pitch_tracking import frames, harmonic_grid, periodicity, track
from src.separator import Analysis, analyze, lpc_evidence, power_strengths, separate, synthesize
from src.visualization import COLORS, save


def frame_figures(split, data, ref, analyses, diagnostics, cfg):
    old, new = analyses
    ac, _ = periodicity(data, cfg)
    both = (ref > 0).all(1)
    available = (np.abs(diagnostics[0]["choices"] / np.maximum(ref[:, :, None], 1)-1) <= .10).any(2)
    recovered = (np.abs(new.tracks.f0 / np.maximum(ref, 1)-1) <= .10).all(1)
    selections = {
        "rejected_pair": both & available.all(1) & ~(old.tracks.f0 > 0).all(1) & recovered,
        "missing_candidate": both & ~available.all(1),
        "remaining_failure": both & ~recovered,
    }
    freq = fft.rfftfreq(cfg.fft, 1/cfg.rate)
    times = np.arange(len(ref))*cfg.hop/cfg.rate
    for reason, subset in selections.items():
        eligible = np.flatnonzero(subset)
        if not eligible.size:
            continue
        i = eligible[len(eligible)//2]
        fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
        fig.suptitle(f"{split} at 0 dB · {reason.replace('_', ' ')} · frame {i} · {times[i]:.2f} s")
        ax = axes.ravel()
        ax[0].plot((np.arange(cfg.window)-cfg.window/2)/cfg.rate*1000, data.samples[i])
        ax[0].set(title="Centered mixture waveform", xlabel="Offset (ms)", ylabel="Full scale")
        magnitude = abs(data.spectrum[i])
        ax[1].plot(freq, 20*np.log10(np.maximum(magnitude/magnitude.max(),1e-5)))
        ax[1].set(title="Mixture spectrum", xlabel="Frequency (Hz)", ylabel="Relative dB", xlim=(0,4000))
        lag = np.arange(ac.shape[1])
        ax[2].plot(1000*lag/cfg.rate, ac[i])
        cep = fft.irfft(np.log(np.maximum(magnitude, magnitude.max()*1e-4+1e-12)), n=cfg.fft)
        q = lag[(1000*lag/cfg.rate >= 2.5) & (1000*lag/cfg.rate <= 14.5)]
        ax[3].plot(1000*q/cfg.rate, cep[q])
        for panel, title, ylabel in [(2,"Normalized autocorrelation","Correlation"),(3,"Real cepstrum","Coefficient")]:
            ax[panel].set(title=title, xlabel="Lag / quefrency (ms)", ylabel=ylabel, xlim=(2.5,14.5))
            for pitch,color in zip(ref[i], COLORS):
                ax[panel].axvline(1000/pitch, color=color, ls="--", lw=1)
        for label, diag in zip(("Original","Improved"),diagnostics):
            ax[4].plot(diag["grid"],diag["harmonic"][i],label=label)
        ax[4].set(title="Harmonic summation on whitened peaks", xlabel="Candidate (Hz)", ylabel="Score")
        ax[4].legend()
        for j, speaker in enumerate(SPEAKERS):
            panel = ax[5+j]
            for label, diag in zip(("Original","Improved"),diagnostics):
                curve = diag["scores"][i,j]
                line, = panel.plot(diag["grid"],curve,label=label)
                choices = diag["choices"][i,j]
                panel.scatter(choices,np.interp(choices,diag["grid"],curve),s=16,color=line.get_color())
            panel.axvline(ref[i,j],ls="--",color=COLORS[j],label="Isolated F0")
            panel.set(title=speaker.capitalize()+" primary score and shortlisted candidates",
                      xlabel="Candidate (Hz)",ylabel="Score")
            panel.legend(fontsize=8)
        nearby = slice(max(0,i-20),min(len(times),i+21))
        for j,speaker in enumerate(SPEAKERS):
            for label,values,marker in [("Isolated",ref,"."),("Original",old.tracks.f0,"+"),("Improved",new.tracks.f0,"x")]:
                q = values[nearby,j]
                keep = q>0
                ax[7].scatter(times[nearby][keep],q[keep],marker=marker,s=12,color=COLORS[j],label=f"{speaker} {label}")
        ax[7].axvline(times[i],ls=":",color="black")
        ax[7].set(title=f"Old {old.tracks.f0[i]} · new {new.tracks.f0[i]} · isolated {ref[i].round(1)} Hz",
                  xlabel="Time (s)",ylabel="F0 (Hz)")
        ax[7].legend(ncol=2,fontsize=7)
        save(fig, DEST / "pitch_diagnostics" / f"{split}_{reason}.png")


def describe(split):
    folder = DEST / split
    p,c = load_profiles(folder/"profiles.json")
    case = folder / "ratio_+0dB"
    x = read_audio(case/"Digi_added_samples.wav")[0]
    refs = np.stack([read_audio(case / f"reference_{s}.wav")[0] for s in SPEAKERS])
    labels,ref = reference_conditions(refs,c)
    data = frames(x,c)
    freq = fft.rfftfreq(c.fft,1/c.rate)
    analyses,diags = [],[]
    for legacy in (True,False):
        diag = {}
        tr = track(data,p,c,legacy=legacy,diagnostics=diag)
        analyses.append(Analysis(data,tr,harmonic_grid(tr.f0,freq,c.rate/c.fft),0))
        diags.append(diag)
    frame_figures(split,data,ref,analyses,diags,c)
    truth = np.stack([abs(frames(y,c).spectrum)**2 for y in refs],axis=1)
    fraction = truth / np.maximum(truth.sum(1,keepdims=True),1e-15)
    fig,axes = plt.subplots(5,1,figsize=(12,11),layout="constrained",sharex=True)
    maps = [("Isolated-source power ratio",fraction[:,0])]
    results = []
    for label,a,mode,method in [("Original",analyses[0],"legacy","harmonic"),
                                ("Normalized power",analyses[1],"power","harmonic"),
                                ("Conservative",analyses[1],"conservative","harmonic"),
                                ("Conservative with LPC",analyses[1],"conservative","harmonic_lpc")]:
        r = separate(a,len(x),p,c,method,mask=mode)
        maps.append((label,r.masks[:,0]))
        for j,speaker in enumerate(SPEAKERS):
            for region,select in [("dominant",fraction[:,j]>.9),
                                  ("shared harmonics",(a.likelihood>.55).all(1) & labels["both_voiced"][:,None]),
                                  ("high frequency",(freq[None]>4000) & labels["high_zcr_unvoiced_overlap"][:,None]),
                                  ("both F0 absent",~(a.tracks.f0>0).any(1)[:,None])]:
                weights = truth[:,j]*select
                results.append(dict(method=label,target=speaker,region=region,
                                    weighted_mean_target_mask=float(np.sum(r.masks[:,j]*weights)/weights.sum()) if weights.sum() else None,
                                    fraction_of_target_power=float(weights.sum()/truth[:,j].sum())))
    for ax,(title,mask) in zip(axes,maps):
        edge = .5*c.rate/c.fft
        ax.imshow(mask.T,origin="lower",aspect="auto",interpolation="nearest",vmin=0,vmax=1,
                  extent=(-.5*c.hop/c.rate,(len(ref)-.5)*c.hop/c.rate,-edge,c.rate/2+edge),cmap="viridis")
        ax.set(title=title,ylabel="Hz",ylim=(0,8000))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{split} · 0 dB · female mask, dark = 0 and yellow = 1")
    save(fig,DEST/"mask_comparison"/f"{split}_female_masks.png")
    write_rows(DEST/"mask_comparison"/f"{split}_allocation.csv",results)
    banks, lpc = [], {}
    band = (freq>=200)&(freq<=4500)
    for speaker in SPEAKERS:
        coefs = np.asarray(p[speaker]["lpc"]["coefficients"])
        env = -np.log(np.maximum(abs(fft.rfft(coefs,n=c.fft)),1e-9))
        env -= env.mean(1,keepdims=True)
        saved = np.asarray(p[speaker]["lpc"]["log_envelopes"])
        bank = saved[:,band].copy()
        bank -= bank.mean(1,keepdims=True)
        bank /= np.linalg.norm(bank,axis=1,keepdims=True)
        banks.append(bank)
        lpc[speaker] = dict(max_pole_radius=max(float(np.max(abs(np.roots(a)))) for a in coefs),
                            max_log_envelope_error=float(np.max(abs(env-saved))),
                            median_within_template_cosine=float(np.median((bank@bank.T)[np.triu_indices(len(bank),1)])))
    lpc["median_cross_template_cosine"] = float(np.median(banks[0]@banks[1].T))
    for label,a in zip(("original","improved"),analyses):
        w = lpc_evidence(a,p,c)
        lpc[label+"_weight_percentiles_5_50_95"] = np.percentile(w,[5,50,95]).tolist()
        lpc[label+"_unity_weight_fraction"] = float(np.mean(w==1))
        present = a.tracks.f0>0
        lpc[label+"_one_frame_dropouts"] = (present[:-2] & ~present[1:-1] & present[2:]).sum(0).tolist()
    write_json(DEST/"mask_comparison"/f"{split}_lpc.json",lpc)


def mask_sensitivity():
    """One-at-a-time controls evaluated only on development waveforms."""
    rows = []
    folder = DEST / "development"
    profiles, cfg = load_profiles(folder / "profiles.json")
    for ratio in RATIOS:
        case = folder / f"ratio_{ratio:+d}dB"
        mix = read_audio(case / "Digi_added_samples.wav")[0]
        refs = np.stack([read_audio(case / f"reference_{s}.wav")[0] for s in SPEAKERS])
        analysis = analyze(mix, profiles, cfg)
        for name, power, smoothing in [("default", 1, (.85,.55)), ("wider_comb", .67, (.85,.55)),
                                       ("narrower_comb", 1.5, (.85,.55)), ("no_time_smoothing", 1, (0,.55)),
                                       ("no_frequency_smoothing", 1, (.85,0)), ("envelope_only", 0, (.85,.55))]:
            modified = replace(analysis, likelihood=analysis.likelihood ** power)
            strengths = power_strengths(modified, profiles, cfg, True)
            result = synthesize(modified, strengths, len(mix), cfg, perf_counter(), smoothing)
            for j, speaker in enumerate(SPEAKERS):
                rows.append(dict(ratio=ratio, variant=name, target=speaker,
                                 **score(result.audio[j].astype(np.float32).astype(float), mix, refs, j)))
    write_rows(DEST / "mask_comparison" / "development_sensitivity.csv", rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("development","validation","test","held_out"))
    parser.add_argument("--mask-sensitivity", action="store_true")
    args = parser.parse_args()
    describe(args.split)
    if args.mask_sensitivity:
        mask_sensitivity()

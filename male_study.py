"""Controlled male-output experiments on the existing, unmodified PCM32 test fixtures."""

import argparse
import csv
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage, signal

from config import FEMALE, MALE, OUTPUT, REAL, SPEAKERS
from male_listening import listening_set
from optimization import write_rows
from src.audio_io import partition, read_audio
from src.calibration import band_starts, create_profiles, load_profiles
from src.cepstrum import log_envelope
from src.evaluation import RATIOS, score
from src.lpc_analysis import fit_lpc
from src.pitch_tracking import frames, inverse_stft, single_pitch
from src.separator import analyze, lpc_evidence, match_band_powers, power_strengths, separate, soft_power_ratio

ROOT = OUTPUT / "male_intelligibility"
HISTORY = OUTPUT / "optimization"
BANDS = ((80, 300), (300, 1000), (1000, 2500), (2500, 4000), (4000, 8000))


def read_case(split, ratio):
    folder = HISTORY / split / f"ratio_{ratio:+d}dB"
    mix = read_audio(folder / "Digi_added_samples.wav")[0]
    refs = np.stack([read_audio(folder / f"reference_{s}.wav")[0] for s in SPEAKERS])
    if not np.array_equal(mix, refs.sum(0)):
        raise ValueError("Reference channels must sum exactly at the mixture sample times")
    return mix, refs


def bin_indices(freq, low, high):
    return (freq >= low) & ((freq <= high) if high == freq[-1] else (freq < high))


def db_ratio(num, den):
    return float(10 * np.log10(max(num, 1e-30) / max(den, 1e-30))) if den > 1e-24 else None


def envelope_difference(output_spec, reference_log, active, cfg):
    freq = fft.rfftfreq(cfg.fft, 1/cfg.rate)
    band = (freq>=80) & (freq<=8000)
    delta = (log_envelope(output_spec[active],cfg,2)-reference_log[active]) * (20/np.log(10))
    offset = delta[:,band].mean(1,keepdims=True)
    return float(np.sqrt(np.mean((delta[:,band]-offset)**2)))


def band_rows(mask, source_power, freq, subsets):
    rows = []
    for low, high in BANDS:
        band = bin_indices(freq, low, high)
        powers = source_power[:, :, band]
        for name, active in subsets.items():
            female, male = powers[:, active]
            m = mask[active][:, band]
            dominant = male > 9*female
            weights = male*dominant
            rows.append(dict(low_hz=low, high_hz=high, subset=name, frames=len(m),
                             male_retained_db=db_ratio(np.sum(male*m**2), male.sum()),
                             female_pass_db=db_ratio(np.sum(female*m**2), female.sum()),
                             male_weighted_mask=float(np.sum(male*m)/male.sum()) if male.sum() else None,
                             male_dominant_mask=float(np.sum(weights*m)/weights.sum()) if weights.sum() else None))
    return rows


def reproduce():
    profiles,cfg = load_profiles(HISTORY/"test"/"profiles.json")
    freq = fft.rfftfreq(cfg.fft,1/cfg.rate)
    metrics, band_metrics = [], []
    fig, axes = plt.subplots(2,1,figsize=(9,7),layout="constrained")
    def draw(ax,name,spectrum):
        power = np.mean(abs(spectrum)**2,axis=0)
        ax.plot(freq,10*np.log10(np.maximum(power,1e-12)),label=name,lw=1)
    for ratio in RATIOS:
        x,refs = read_case("test",ratio)
        a = analyze(x,profiles,cfg)
        result = separate(a,len(x),profiles,cfg,"harmonic_lpc")
        for i,name in enumerate(SPEAKERS):
            y = result.audio[i].astype(np.float32).astype(float)
            path = HISTORY/"test"/f"ratio_{ratio:+d}dB"/f"output_{name}.wav"
            assert np.array_equal(y,read_audio(path)[0])
            metrics.append(dict(ratio=ratio,speaker=name,**score(y,x,refs,i)))
        spectra = np.stack([frames(v,cfg).spectrum for v in refs])
        for row in band_rows(result.masks[:,1],abs(spectra)**2,freq,{"all":slice(None)}):
            band_metrics.append(dict(ratio=ratio,lo=row["low_hz"],hi=row["high_hz"],
                                     male_retained_db=row["male_retained_db"],female_pass_db=row["female_pass_db"],
                                     dominant_male_mask=row["male_dominant_mask"]))
        if ratio==0:
            for i,name in enumerate(SPEAKERS):
                draw(axes[0],name+" reference",spectra[i])
                draw(axes[0],name+" output",frames(result.audio[i],cfg).spectrum)
            draw(axes[0],"Mixture",a.frames.spectrum)
    for name in ("mixture","male","female"):
        file = HISTORY/"final_cli"/"real"/"harmonic_lpc"/f"output_{name}.wav"
        draw(axes[1],name,frames(read_audio(file)[0],cfg).spectrum)
    for ax,title in zip(axes,("Strict 0 dB mixture","Physical microphone recording")):
        ax.set(xlabel="Frequency (Hz)",ylabel="Mean FFT power (dB)",title=title,xscale="log",xlim=(80,8000))
        ax.legend()
    fig.savefig(ROOT/"diagnosis"/"baseline_spectra.png",dpi=140)
    plt.close(fig)
    write_rows(ROOT/"diagnosis"/"baseline_reproduction.csv",metrics)
    write_rows(ROOT/"diagnosis"/"baseline_bands.csv",band_metrics)


def diagnose():
    """Measure contributions by applying the mixture-derived mask to each isolated STFT."""
    split = "test"
    profiles, cfg = load_profiles(HISTORY/split/"profiles.json")
    x, refs = read_case(split, 0)
    a = analyze(x, profiles, cfg)
    r = separate(a, len(x), profiles, cfg, "harmonic_lpc")
    female, male = [frames(y, cfg) for y in refs]
    Y = frames(r.audio[1], cfg).spectrum
    freq = fft.rfftfreq(cfg.fft, 1/cfg.rate)
    voiced = single_pitch(male, cfg)[0] > 0
    subsets = dict(all=np.ones_like(voiced), male_voiced=voiced, male_active_unvoiced=male.active & ~voiced,
                   male_high_zcr=male.active & (male.zcr >= cfg.unvoiced_zcr) & ~voiced)
    centered_power = np.abs(np.stack([female.spectrum, male.spectrum]))**2
    base = power_strengths(a, profiles, cfg, True)
    weighted = base * lpc_evidence(a, profiles, cfg)
    raw = weighted[:,0]/weighted.sum(1)
    # Turning comb modulation off retains the fitted spectral-power envelope and uncertainty floor.
    unmodulated = replace(a, likelihood=np.ones_like(a.likelihood))
    power = power_strengths(unmodulated, profiles, cfg, True)
    masks = dict(baseline=r.masks[:,1], power_only=power[:,1]/power.sum(1),
                 before_lpc=base[:,1]/base.sum(1), unsmoothed=1-raw,
                 frequency_only=1-ndimage.gaussian_filter(raw, (0,.55), mode="nearest"),
                 time_only=1-ndimage.gaussian_filter(raw, (.85,0), mode="nearest"))
    rows = [dict(stage=name, **row) for name,m in masks.items()
            for row in band_rows(m, centered_power, freq, subsets)]
    write_rows(ROOT/"diagnosis"/"components.csv", rows)
    formants = []
    male_mask = r.masks[:,1]
    for i in np.flatnonzero(voiced):
        _, _, resonances = fit_lpc(male.samples[i], cfg)
        for rank, f in enumerate(resonances[:3], 1):
            near = abs(freq-f) <= 100
            source = centered_power[1,i,near]
            formants.append(dict(frame=int(i), formant=rank, hz=f, source_power=source.sum(),
                                 passed_power=np.sum(source*male_mask[i,near]**2)))
    write_rows(ROOT/"diagnosis"/"male_lpc_resonances.csv", formants)
    observed = []
    for low,high in BANDS:
        b = bin_indices(freq,low,high)
        for tag,s in [('male_output',Y),('female_output',frames(r.audio[0],cfg).spectrum),
                      ('mixture',a.frames.spectrum),('female_reference',female.spectrum),('male_reference',male.spectrum)]:
            ref = abs(male.spectrum[ male.active][:,b])
            observed.append(dict(signal=tag,low_hz=low,high_hz=high,
                                 rms_magnitude_ratio_db=db_ratio(np.sum(abs(s[male.active][:,b])**2),np.sum(ref**2)),
                                 mean_magnitude_ratio=float(abs(s[male.active][:,b]).sum()/ref.sum())))
    write_rows(ROOT/"diagnosis"/"observed_spectra.csv",observed)
    components = np.stack([inverse_stft(d.spectrum*male_mask,len(x),cfg) for d in (female,male)])
    assert np.max(abs(components.sum(0)-r.audio[1])) < 1e-12
    rendered = np.stack([abs(frames(v,cfg).spectrum)**2 for v in components])
    recovered = []
    for low,high in BANDS:
        b = bin_indices(freq,low,high)
        passed = rendered[:,:,b].sum((1,2))
        original = centered_power[:,:,b].sum((1,2))
        recovered.append(dict(low_hz=low,high_hz=high,rendered_male_pass_db=db_ratio(passed[1],original[1]),
                              rendered_female_pass_db=db_ratio(passed[0],original[0]),
                              female_over_male_output_db=db_ratio(passed[0],passed[1])))
    write_rows(ROOT/"diagnosis"/"rendered_components.csv",recovered)
    reference_log = log_envelope(male.spectrum,cfg,2.)
    delta = 20/np.log(10)*(log_envelope(Y,cfg,2.)-reference_log)
    envelope_rows = []
    for name,selected in subsets.items():
        for low,high in BANDS:
            within = bin_indices(freq,low,high)
            d = delta[selected][:,within]
            envelope_rows.append(dict(subset=name,low_hz=low,high_hz=high,
                                      median_observed_minus_male_db=np.median(d),rmse_db=np.sqrt(np.mean(d**2))))
    write_rows(ROOT/"diagnosis"/"observed_envelope_difference.csv",envelope_rows)
    real_dir = HISTORY/"final_cli"/"real"/"harmonic_lpc"
    distribution = []
    for name in ("mixture","male","female"):
        real_audio = read_audio(real_dir/f"output_{name}.wav")[0]
        power = np.mean(abs(frames(real_audio,cfg).spectrum)**2,axis=0)
        if name == "mixture":
            real_mix = power
        for low,high in BANDS:
            b = bin_indices(freq,low,high)
            distribution.append(dict(output=name,low_hz=low,high_hz=high,
                                     power_share=power[b].sum()/power.sum(),
                                     relative_to_mixture_db=db_ratio(power[b].sum(),real_mix[b].sum())))
    write_rows(ROOT/"diagnosis"/"real_output_distribution.csv",distribution)
    # Two factorial phase controls. All four are evaluation-only except baseline.
    rows = []
    for magnitude, mag in [('estimated', abs(a.frames.spectrum)*male_mask), ('reference',abs(male.spectrum))]:
        for name, spec in [('mixture',a.frames.spectrum),('reference',male.spectrum)]:
            y = inverse_stft(mag*np.exp(1j*np.angle(spec)),len(x),cfg)
            rows.append(dict(magnitude=magnitude,phase=name,**score(y,x,refs,1)))
    write_rows(ROOT/"diagnosis"/"phase_controls.csv",rows)


def cepstral_diagnosis():
    """Compare shape matches on disjoint development frames before trying a mask."""
    from scipy.spatial.distance import cdist
    p, cfg = load_profiles(HISTORY/"development"/"profiles.json")
    x, refs = read_case("development",0)
    a = analyze(x,p,cfg)
    y = separate(a,len(x),p,cfg,"harmonic_lpc").audio[1]
    starts = band_starts(cfg)
    def features(spec, cutoff):
        power = abs(spec)**2 if cutoff == 0 else np.exp(2*log_envelope(spec,cfg,cutoff))
        v = np.log(np.add.reduceat(power,starts,axis=1)+1e-12)
        return v-v.mean(1,keepdims=True)
    banks = {cutoff:[] for cutoff in (0,1,2)}
    for path in (FEMALE,MALE):
        audio,_,_ = partition(path,cfg)
        d = frames(audio,cfg)
        voiced = single_pitch(d,cfg)[0] > 0
        picks = []
        for group in (voiced,d.active & ~voiced):
            ix = np.flatnonzero(group)
            if ix.size:
                picks.extend(ix[np.linspace(0,len(ix)-1,min(16,len(ix)),dtype=int)])
        for cutoff in banks:
            banks[cutoff].append(features(d.spectrum[picks],cutoff))
    rows = []
    for j,speaker in enumerate(SPEAKERS):
        d = frames(refs[j],cfg)
        voiced = single_pitch(d,cfg)[0]>0
        predictions = {}
        for cutoff in banks:
            obs = features(d.spectrum,cutoff)
            distances = [cdist(obs,bank).min(1) for bank in banks[cutoff]]
            predictions[cutoff] = np.argmin(distances,axis=0)
        for subset, active in [('voiced',voiced),('active_unvoiced',d.active & ~voiced)]:
            for cutoff,predicted in predictions.items():
                correct = predicted[active]==j
                raw_ok = predictions[0][active]==j
                rows.append(dict(speaker=speaker,subset=subset,cutoff_ms=cutoff,frames=int(active.sum()),
                                 correct_fraction=correct.mean(),gained_vs_raw=int((correct & ~raw_ok).sum()),
                                 lost_vs_raw=int((~correct & raw_ok).sum())))
    write_rows(ROOT/"diagnosis"/"cepstral_matching.csv",rows)
    # Direct development probe in the mixture. Fit powers of liftered envelopes
    # from calibration examples. The fit uses no isolated development signal.
    baseline = separate(a,len(x),p,cfg,"harmonic_lpc")
    probe = []
    male_data = frames(refs[1],cfg)
    reference_log = log_envelope(male_data.spectrum,cfg,2)
    freq = fft.rfftfreq(cfg.fft,1/cfg.rate)
    upper = (freq>=2500) & (freq<4000)
    target_power = abs(male_data.spectrum[:,upper])**2
    def probe_row(label, estimate, mask):
        return dict(cutoff_ms=label, **score(estimate,x,refs,1),
                    envelope_shape_rmse_db=envelope_difference(frames(estimate,cfg).spectrum,reference_log,male_data.active,cfg),
                    upper_male_retained_db=db_ratio(np.sum(target_power*mask[:,upper]**2),target_power.sum()))
    for cutoff in (1,2):
        normalized = []
        for bank in banks[cutoff]:
            power = np.exp(bank-bank.max(1,keepdims=True))
            normalized.append(power/power.sum(1,keepdims=True))
        mix_power = np.add.reduceat(np.exp(2*log_envelope(a.frames.spectrum,cfg,cutoff)),starts,axis=1)
        bands, fit = match_band_powers(mix_power,normalized)
        ratio = soft_power_ratio(bands[:,0],bands[:,1],1)
        centers = starts+(np.diff(np.r_[starts,cfg.fft//2+1])-1)/2
        smooth = np.stack([np.interp(np.arange(cfg.fft//2+1),centers,row) for row in ratio])
        mask = baseline.masks[:,1] + .35*fit[:,None]*(smooth-baseline.masks[:,1])
        y_probe = inverse_stft(a.frames.spectrum*mask,len(x),cfg)
        probe.append(probe_row(cutoff,y_probe,mask))
    for name,opts in [('A',{}),('B',{'male_variant':'power_only'})]:
        result = separate(a,len(x),p,cfg,"harmonic_lpc",**opts)
        probe.append(probe_row(name,result.audio[1],result.masks[:,1]))
    write_rows(ROOT/"diagnosis"/"cepstral_development_probe.csv",probe)
    freq = fft.rfftfreq(cfg.fft,1/cfg.rate)
    ref_frames = frames(refs[1],cfg)
    voiced = single_pitch(ref_frames,cfg)[0]>0
    fig,axes = plt.subplots(2,2,figsize=(12,7),layout="constrained",sharex=True)
    for j,(label,sel) in enumerate([("Male voiced",voiced),("Male active unvoiced",ref_frames.active & ~voiced)]):
        ixs = np.flatnonzero(sel)
        i = ixs[len(ixs)//2]
        for k,cutoff in enumerate((1,2)):
            ax = axes[j,k]
            for name,sig in [('Male reference',refs[1]),('Female reference',refs[0]),('Mixture',x),('Male A',y)]:
                env = log_envelope(frames(sig,cfg).spectrum[i],cfg,cutoff)
                ax.plot(freq,env*20/np.log(10),label=name)
            ax.set(title=f"{label}, {i*cfg.hop/cfg.rate:.2f} s, cutoff {cutoff} ms",
                   xlabel="Frequency (Hz)",ylabel="Log magnitude (dB)",ylim=(-80,40))
            ax.legend(fontsize=8)
    fig.savefig(ROOT/"diagnosis"/"cepstral_envelopes.png",dpi=145)
    plt.close(fig)


VARIANTS = {
    "A_current": {}, "B_power_only": {"male_variant": "power_only"},
    "C_reduced_harmonic": {"male_variant": "reduced_harmonic"},
    "D_reduced_smoothing": {"male_variant": "reduced_smoothing"},
    "E_conservative_mask": {"male_variant": "conservative_mask"},
    "G_power_ratio": {"male_variant": "power_ratio"},
    "G_ratio_055": {"male_variant": "power_ratio", "power_exponent": .55},
    "C_confidence": {"male_variant": "half_confidence"}, "C_width": {"male_variant": "wider_comb"},
    "D_time": {"male_variant": "less_time"}, "D_frequency": {"male_variant": "less_frequency"},
    "D_band_time": {"male_variant": "less_band_smoothing"},
    "F_cepstral_assisted": {"male_variant": "cepstral_assisted"},
    "H_presence_gain": {"male_variant": "presence_gain"},
}


def calibrated_profiles(split, cfg, window):
    path = ROOT/"comparisons"/"profiles"/f"{split}_{window}.json"
    settings = replace(cfg, window=window)
    create_profiles((FEMALE,MALE),path,settings,include_cepstral=(window==cfg.window))
    return load_profiles(path)[0]


def study(split, pilot=False):
    profiles, cfg = load_profiles(HISTORY/split/"profiles.json")
    alternate = {}
    if not pilot:
        enhanced = calibrated_profiles(split,cfg,cfg.window)
        for speaker in SPEAKERS:
            assert enhanced[speaker]["recording"] == profiles[speaker]["recording"]
            profiles[speaker]["cepstral_2ms_power_examples"] = enhanced[speaker]["cepstral_2ms_power_examples"]
        for window in (512,1024):
            alternate[window] = calibrated_profiles(split,cfg,window)
    metrics, bands = [], []
    for ratio in RATIOS:
        x, refs = read_case(split, ratio)
        a = analyze(x, profiles, cfg)
        src = [frames(y,cfg) for y in refs]
        powers = np.abs(np.stack([s.spectrum for s in src]))**2
        voiced = single_pitch(src[1],cfg)[0] > 0
        subsets = dict(all=np.ones_like(voiced), male_voiced=voiced, male_unvoiced=src[1].active & ~voiced)
        reference_log = log_envelope(src[1].spectrum,cfg,2.)
        female_baseline = None
        variants = dict(VARIANTS)
        if pilot:
            for name in ("F_cepstral_assisted","H_presence_gain"):
                del variants[name]
        variants.update({f"W_{n//16}ms": {"window":n} for n in alternate})
        for variant, options in variants.items():
            local_cfg, local_powers = cfg, powers
            if "window" in options:
                window = options["window"]
                local_cfg = replace(cfg,window=window)
                other = analyze(x,alternate[window],local_cfg)
                r = separate(other,len(x),alternate[window],local_cfg,"harmonic_lpc")
                times = baseline_seconds + other.seconds + r.seconds
                r.audio[0] = female_baseline
                local_powers = np.stack([abs(frames(v,local_cfg).spectrum)**2 for v in refs])
                assert np.max(abs(inverse_stft(other.frames.spectrum,len(x),local_cfg)-x))<1e-12
            else:
                r = separate(a, len(x), profiles, cfg, "harmonic_lpc", **options)
                times = a.seconds+r.seconds
            if female_baseline is None:
                female_baseline = r.audio[0].copy()
                female_score = score(female_baseline.astype(np.float32).astype(float), x, refs, 0)
                baseline_seconds = times
            assert np.array_equal(female_baseline, r.audio[0])
            assert r.audio.shape == refs.shape and np.isfinite(r.audio).all()
            y = r.audio[1].astype(np.float32).astype(float)
            key = dict(split=split, ratio=ratio, variant=variant)
            metrics.append(dict(**key, **score(y,x,refs,1), female_si_sdri_db=female_score["si_sdri_db"],
                                envelope_shape_rmse_db=envelope_difference(frames(y,cfg).spectrum,
                                                                          reference_log,src[1].active,cfg),
                                seconds=times,rtf=times/(len(x)/cfg.rate),
                                mixture_sum_max_error=np.max(abs(r.audio.sum(0)-x))))
            bands.extend(dict(**key, **row) for row in band_rows(r.masks[:,1],local_powers,
                         fft.rfftfreq(local_cfg.fft,1/local_cfg.rate),subsets))
        print(f"{split} {ratio:+} dB", flush=True)
    stem = ("pilot_" if pilot else "") + split
    write_rows(ROOT/"comparisons"/f"{stem}_scores.csv",metrics)
    write_rows(ROOT/"comparisons"/f"{stem}_bands.csv",bands)


def listening():
    profile_path = ROOT/"comparisons"/"profiles"/"test_640.json"
    profiles,cfg = load_profiles(profile_path)
    x,refs = read_case("test",0)
    for label,mix in [("controlled_0dB",x),("real",read_audio(REAL)[0])]:
        a = analyze(mix,profiles,cfg)
        signals = {"original_mixture":mix}
        if label=="controlled_0dB":
            signals["reference_male"] = refs[1]
        times = []
        for variant in ("A_current","B_power_only","C_reduced_harmonic","D_reduced_smoothing",
                        "E_conservative_mask","F_cepstral_assisted","G_power_ratio","H_presence_gain"):
            r = separate(a,len(mix),profiles,cfg,"harmonic_lpc",**VARIANTS[variant])
            if variant=="A_current":
                signals["female_original_optimized"] = r.audio[0]
                baseline_dir = (HISTORY/"test"/"ratio_+0dB" if label=="controlled_0dB"
                                else HISTORY/"final_cli"/"real"/"harmonic_lpc")
                for j,s in enumerate(SPEAKERS):
                    assert np.array_equal(read_audio(baseline_dir/f"output_{s}.wav")[0],r.audio[j].astype(np.float32))
            assert np.array_equal(signals["female_original_optimized"],r.audio[0])
            signals["male_"+variant] = r.audio[1]
            times.append(dict(variant=variant,seconds=a.seconds+r.seconds,rtf=(a.seconds+r.seconds)/(len(mix)/cfg.rate)))
        for window in (512,1024):
            other,config = load_profiles(profile_path.with_name(f"test_{window}.json"))
            alt = analyze(mix,other,config)
            r = separate(alt,len(mix),other,config,"harmonic_lpc")
            signals[f"male_W_{window//16}ms"] = r.audio[1]
            seconds = times[0]["seconds"]+alt.seconds+r.seconds
            times.append(dict(variant=f"W_{window//16}ms",seconds=seconds,rtf=seconds/(len(mix)/cfg.rate)))
        folder = ROOT/"listening"/label
        listening_set(folder,signals,cfg.rate)
        write_rows(folder/"runtime.csv",times)


def collect():
    def read(path):
        with path.open(newline="") as file:
            return list(csv.DictReader(file))
    metrics, bands = [], []
    for split in ("development","validation","test"):
        metrics.extend(read(ROOT/"comparisons"/f"{split}_scores.csv"))
        bands.extend(read(ROOT/"comparisons"/f"{split}_bands.csv"))
    write_rows(ROOT/"metrics.csv",metrics)
    summaries = []
    for split,variant in dict.fromkeys((r["split"],r["variant"]) for r in metrics):
        selected = [r for r in metrics if r["split"]==split and r["variant"]==variant]
        row = dict(split=split,variant=variant)
        for k in ("si_sdri_db","target_gain_db","interferer_attenuation_db","envelope_shape_rmse_db","rtf"):
            row[k] = np.mean([float(r[k]) for r in selected])
        for low,_ in BANDS:
            values = [float(r["male_retained_db"]) for r in bands if r["split"]==split and r["variant"]==variant
                      and r["subset"]=="all" and int(r["low_hz"])==low]
            row[f"retained_from_{low}_db"] = np.mean(values)
        summaries.append(row)
    write_rows(ROOT/"experiment_results.csv",summaries)
    chosen = ["A_current","B_power_only","D_reduced_smoothing","E_conservative_mask","F_cepstral_assisted","H_presence_gain"]
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for name in chosen:
        r = next(v for v in summaries if v["split"]=="test" and v["variant"]==name)
        line, = axes[0].plot(range(5),[r[f"retained_from_{low}_db"] for low,_ in BANDS],marker="o",label=name[0])
        axes[1].scatter(r["interferer_attenuation_db"],r["retained_from_2500_db"],color=line.get_color())
        axes[1].annotate(name[0],(r["interferer_attenuation_db"],r["retained_from_2500_db"]),
                         xytext=(4,4),textcoords="offset points")
    axes[0].set(xticks=range(5),xticklabels=["80–300","300–1k","1–2.5k","2.5–4k","4–8k"],
                xlabel="Frequency band (Hz)",ylabel="Retained male power (dB)",title="Strict test, mean of five ratios")
    axes[0].legend(ncol=3)
    axes[1].set(xlabel="Female interferer attenuation (dB)",ylabel="Male power retained at 2.5–4 kHz (dB)",
                title="Target detail and leakage")
    fig.savefig(ROOT/"comparisons"/"preservation_and_leakage.png",dpi=140)
    plt.close(fig)


def window_diagnostics():
    """Compare window-dependent tracking and male transient envelopes on one fixed time grid."""
    mix,refs = read_case("test",0)
    _,base = load_profiles(HISTORY/"test"/"profiles.json")
    truth = np.stack([single_pitch(frames(v,base),base)[0] for v in refs],axis=1)
    both = (truth>0).all(1)
    filt = signal.butter(4,2500,fs=base.rate,btype="highpass",output="sos")
    def highband_db(y):
        y = signal.sosfiltfilt(filt,y)
        blocks = y[:len(y)//base.hop*base.hop].reshape(-1,base.hop)
        return 10*np.log10(np.mean(blocks**2,axis=1)+1e-14)
    reference_level = highband_db(refs[1])
    rise = np.r_[0., 0., reference_level[2:]-reference_level[:-2]]
    peaks = signal.find_peaks(rise,height=6,distance=5)[0]
    peaks = peaks[(peaks>=2) & (reference_level[peaks]>reference_level.max()-35)]
    rows=[]
    for n in (512,640,1024):
        p,c=load_profiles(ROOT/"comparisons"/"profiles"/f"test_{n}.json")
        a=analyze(mix,p,c)
        r=separate(a,len(mix),p,c,"harmonic_lpc")
        component=inverse_stft(frames(refs[1],c).spectrum*r.masks[:,1],len(mix),c)
        level=highband_db(component)
        rows.append(dict(window_ms=n//16,both_voiced_frames=int(both.sum()),
                         correct_pair_fraction=(abs(a.tracks.f0[both]/truth[both]-1)<=.1).all(1).mean(),
                         male_voiced_accuracy=(abs(a.tracks.f0[:,1][truth[:,1]>0]/truth[:,1][truth[:,1]>0]-1)<=.1).mean(),
                         highband_attacks=len(peaks),median_reference_20ms_rise_db=np.median(rise[peaks]),
                         median_output_20ms_rise_db=np.median(level[peaks]-level[peaks-2]),
                         median_absolute_rise_error_db=np.median(abs(level[peaks]-level[peaks-2]-rise[peaks]))))
    write_rows(ROOT/"diagnosis"/"window_attacks_and_pitch.csv",rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("reproduce","diagnose","cepstrum","pilot","development","validation","test","listening","collect","windows"))
    stage = parser.parse_args().stage
    if stage == "diagnose":
        diagnose()
    elif stage == "reproduce":
        reproduce()
    elif stage == "cepstrum":
        cepstral_diagnosis()
    elif stage == "listening":
        listening()
    elif stage == "collect":
        collect()
    elif stage == "windows":
        window_diagnostics()
    elif stage == "pilot":
        study("development",pilot=True)
    else:
        study(stage)

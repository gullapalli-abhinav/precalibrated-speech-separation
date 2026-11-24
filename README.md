# Pre-Calibrated Single-Channel Speech Separation Using Classical DSP

**Version:** v0.1.0-offline
**Stage:** Pre-calibrated offline prototype
**Processing:** Recorded audio, non-real-time

Individual speaker recordings are collected beforehand for calibration. The system accepts prerecorded single-channel mixed speech and writes individual estimates and a selectable mute output. Live microphone streaming is not implemented. Centered frames and offline trajectory tracking can use future audio. Faster-than-real-time throughput does not make this a real-time system. Streaming selective muting is a future milestone.

An offline Python prototype for two calibrated speakers sharing one microphone. Given one mono mixture, it generates estimates of each speaker, a mixture copy, and a selectable output. It uses classical DSP throughout.

## Optimized separator

The implementation uses STFT, normalized autocorrelation, real-cepstral pitch evidence, harmonic analysis, LPC envelopes and formant diagnostics, calibrated spectral-power estimation, complementary time-frequency masks, and inverse STFT reconstruction. Calibration must use previously recorded individual speaker samples. The CLI processes WAV files offline.

The [male intelligibility study](outputs/male_intelligibility/diagnosis_report.md) compares independent male mask choices and analysis windows. The [listening guide](outputs/male_intelligibility/recommendation.md) links loudness-matched physical and controlled comparisons. Normal separation still uses the original 40 ms optimized pair.

For the 64 ms male candidate, supply its profile as a second pass. For the cepstral candidate, select a profile containing the optional 2 ms envelope examples. Each command preserves the baseline female output and writes to its own directory.

```powershell
python main.py separate --input test.wav --profiles outputs/profiles.json --male-window-profiles outputs/male_intelligibility/comparisons/profiles/test_1024.json --mode mute-female --out-dir outputs/male_64ms
python main.py separate --input test.wav --profiles outputs/male_intelligibility/comparisons/profiles/test_640.json --male-variant cepstral_assisted --mode mute-female --out-dir outputs/male_cepstral
```

For new recordings, `calibrate --window-ms 64` creates the alternate window profile and `calibrate --cepstral` adds the optional cepstral bank to a 40 ms profile. Use distinct output directories and the same calibration sources for both windows. `separate --help` lists male mask variants and their parameters. Independent male estimates can overlap the protected female estimate, so their sum can differ from the mixture.

The default uses refined joint pitch tracking, a normalized source-power mask and a bounded LPC adjustment (`harmonic_lpc`). Calibration saves short-time band-power examples from both voiced and unvoiced speech. The separator matches pairs of those fixed examples to the mixture to estimate how much energy each speaker contributes in each frequency band. This gives it a way to assign consonants and shared harmonics. Both outputs use the mixture phase and sum to the input. The `harmonic` option uses the same power mask without LPC.

Existing version-1 profiles still load. Running `calibrate` adds the optional band-power examples to a new profile. A profile without examples uses the pitch-comb fallback. The original implementation and fixed historical profile are preserved in the [baseline snapshot](outputs/optimization/baseline_snapshot.zip).

The [optimization report](outputs/optimization/optimization_report.md) gives the strict held-out scores, the full historical comparison, pitch diagnostics, oracle controls, timing and reproduction commands.

## Run the project

Use your own local recordings with the documented filenames `female_voice_sample.waw.wav`, `male_voice_sample.wav`, `test.wav`, and optional `Digi_added_samples.wav`. Voice recordings, calibration profiles and generated outputs are ignored by Git. Profiles contain voice-derived information and should remain private.

Use Python 3.11 or later. Run these commands from the project directory.

```powershell
python -m pip install -r requirements.txt
python main.py inspect
python main.py calibrate
python main.py evaluate
python -m unittest discover -s tests -v
```

The four supplied WAVs are at the project root. Edit `config.py` or pass `--female`, `--male`, and `--input` if using other recordings. A WAV is decoded from its header, so an alternate extension can be passed explicitly. WAV inputs with multiple channels are averaged to mono and converted to 16 kHz by polyphase resampling. The separator accepts a single resulting waveform.

```powershell
python main.py separate --input test.wav --mode mute-male --method harmonic
python main.py separate --input another_mix.wav --mode mute-female --profiles outputs/profiles.json --out-dir outputs/another_mix
python main.py visualize --input test.wav --method harmonic_lpc --profiles outputs/profiles.json --out-dir outputs/visualization
```

`mute-male` selects the female estimate, `mute-female` selects the male estimate, `both` selects the input and `mute-both` writes zeros. `separate` saves `output_female.wav`, `output_male.wav`, `output_mixture.wav`, `output_selected.wav`, a pitch CSV, and diagnostic JSON. The same profiles can be loaded for any new mixture of the two calibrated speakers. Use `python main.py <command> --help` for paths and options. Both variants use the same pitch tracks. The default is `harmonic_lpc`.

## Recordings

All supplied files are 16 kHz mono PCM16 WAVs. Peak and RMS are linear full-scale fractions. A rail sample is at one of the PCM16 limits.

| Recording | Role | Duration (s) | Peak | RMS | Rail samples |
| --- | --- | ---: | ---: | ---: | ---: |
| `female_voice_sample.waw.wav` | Isolated female | 64.024 | 1.00000 | 0.14534 | 1 |
| `male_voice_sample.wav` | Isolated male | 68.994 | 0.99371 | 0.17429 | 0 |
| `test.wav` | Physical shared-microphone recording | 67.033 | 1.00000 | 0.16849 | 1 |
| `Digi_added_samples.wav` | Supplied digital sum | 68.994 | 1.00000 | 0.22280 | 908 |

Whole-file RMS ranges from 0.145 to 0.223 full scale, with quiet pauses in the recordings. The supplied digital waveform matches the PCM16-saturated samplewise sum of the **full** isolated tracks, with zeros after the shorter track. The sum exceeds the PCM16 range at exactly 908 positions. It includes calibration audio and the saturated samples differ from a linear sum, so it has its own reference-free diagnostic run. The originals stay at the project root.

## Held-out mixture generation

The first 60% of each isolated track is used for calibration. A 0.25 s guard follows, and the remainder is held out. Calibration and held-out regions are resampled independently if rate conversion is needed, which prevents a filter from sharing native samples across splits. Each side of the split must be at least two seconds long.

| Source | Calibration samples | Held-out samples | Held-out samples used per ratio |
| --- | --- | --- | --- |
| Female | `[0, 614628)` | `[618628, 1024380)` | `[618628, 1024380)` |
| Male | `[0, 662340)` | `[666340, 1103900)` | `[666340, 1072092)` |

Sample intervals are half-open and at 16 kHz. The five ratio runs reuse the same 25.3595 s excerpts to isolate the effect of level. A sixth paired run delays male speech by two seconds and pads with silence to vary overlap. Female-only and male-only runs use an exactly silent second reference. Each run records origin ranges and time offsets in its `mixture.json`.

For a requested female-to-male RMS ratio `R` in dB, gain the female source to RMS `0.10 * 10^(R/40)` and the male source to RMS `0.10 * 10^(-R/40)`. The RMS calculation includes silence. Multiply both by the same gain if either reference or their sum would exceed the −1 dBFS peak ceiling. Round the two references to signed PCM32, sum them in 64-bit integer arithmetic, and save the sum as PCM32. One reference sample has at most half a PCM32 step of rounding error, about `2.33e-10` in full-scale units. The saved mixture equals the sum of the saved references **exactly**, including on readback in float64.

The output directories contain a generated `Digi_added_samples.wav`, `reference_female.wav`, and `reference_male.wav` per condition. The original file of the same name stays at the project root. Estimates and playback copies use float32 WAV. The exact-sum guarantee applies to each PCM32 triplet. The float32 playback copies and estimates have their own conversion rounding.

The calibration JSON records source SHA-256s, split ranges, DSP settings, and fixed per-speaker statistics. Evaluation checks source identity and split geometry against that file, loads the profiles once, and checks the profile digest again at the end. The held-out references are available to generation and scoring. At inference the separator receives the mixture waveform and the fixed profiles.

## Original DSP implementation (baseline)

1. **Calibration.** Measure short-time energy (STE) and zero-crossing rate (ZCR) with 40 ms Hann frames stepped every 10 ms. Normalized autocorrelation compares overlapping frame slices at each lag, with normalization by both slice energies. Pick the earliest sufficiently strong peak near the best correlation and interpolate its lag. A real cepstrum from the inverse transform of log magnitude gives a separate period diagnostic. Save voiced pitch percentiles, a histogram, confidence, relative harmonic amplitudes, LPC coefficient sets, spectral envelopes and estimated formants. The profile's measured F0 distribution sets each speaker's prior.
2. **Mixture pitch tracking.** Score 70–350 Hz candidates using whitened spectral peaks, autocorrelation and cepstral evidence. For each pair, attenuate spectral peaks explained by the first candidate when checking evidence for the second. Evaluate both orders. The joint search includes unpitched states, discourages double use of similar pitches and uses dynamic programming to penalize jumps and rapid switches. Confidence measures the selected state's frame evidence against a state with that pitch removed. Pitch confidence is a heuristic score. STE gates silent frames and ZCR adds evidence for likely unvoiced regions.
3. **Harmonic masks.** A 1024-point real STFT uses 15.625 Hz bins. For each estimate, compare a bin frequency with the nearest F0 multiple using a Gaussian comb that widens with frequency. Combine powered comb strengths and pitch confidence with a configurable fallback share, then smooth the female mask along time and frequency. The male mask is its complement. Inverse FFT uses the observed mixture phase and Hann-weighted overlap-add with squared-window normalization. Outputs retain the original 16 kHz duration and their float64 sum reconstructs the working mixture.
4. **LPC-assisted masks.** An order-14 autocorrelation LPC model represents vocal-tract resonances. Compare a smoothed spectral shape against several templates for each speaker. The closest templates apply a bounded, confidence-weighted adjustment to harmonic strengths. Phonetic variation, overlapping speech and shared resonances limit how reliably these templates identify a voice.

Normalized period evidence and harmonic summation are described by [de Cheveigné and Kawahara](https://pubmed.ncbi.nlm.nih.gov/12002874/) and [Klapuri](https://www.ee.columbia.edu/~dpwe/papers/Klap06-multif0.pdf).

## Evaluation metrics

- **SI-SDR.** Center estimate and target. Project the estimate onto the target using `a = dot(estimate, target) / dot(target, target)`. Report `10 log10(energy(a * target) / energy(estimate - a * target))`. SI-SDR improvement subtracts the input mixture score. Scores are bounded to ±120 dB for display. A silent reference or zero estimate yields an empty SI-SDR field. Improvement is also empty when the mixture baseline reaches the +120 dB bound, including single-source cases. [Le Roux et al., SI-SDR](https://merl.com/publications/docs/TR2019-013.pdf)
- **Target gain and interferer attenuation.** Regress each centered estimate on the available centered references jointly. Target gain is `20 log10(abs(target coefficient))` and attenuation is the negative of the same calculation on the interferer coefficient. These measure source-correlated components. SI-SDR also reflects distortions outside that projection. For an absent target the output-to-input RMS measures leakage into its output.
- **Subsets and F0.** Reference STE and isolated F0 estimates mark simultaneous activity, one-active frames, both-voiced frames (a vowel proxy), F0 intervals and high-ZCR unvoiced overlap (a consonant-rich proxy). The CSV scores subsets with at least 0.25 s of assigned samples. The JSON lists all subset durations. F0 accuracy compares estimates within 10% of the isolated autocorrelation F0, which is itself a proxy for true F0.
- **Performance.** Processing time includes the mixture STFT, pitch tracking, likelihood maps, mask synthesis and inverse STFT. It excludes calibration, file I/O and plotting. Real-time factor is processing seconds divided by input seconds.

## Historical artifacts

- [Evaluation report](outputs/evaluation.json) and [metrics CSV](outputs/metrics.csv)
- [Original fixed profiles](outputs/optimization/reproduction/profiles.json), [female calibration](outputs/calibration/female_calibration.png) and [male calibration](outputs/calibration/male_calibration.png)
- [Ratio comparison plot](outputs/method_comparison.png) and [controlled runs](outputs/controlled)
- [Real recording tracks](outputs/real/harmonic/mixture_and_pitch.png) and [real masks and output spectrograms](outputs/real/harmonic/masks_and_outputs.png)
- [Real female estimate](outputs/real/harmonic/output_female.wav), [male estimate](outputs/real/harmonic/output_male.wav), [mixture copy](outputs/real/harmonic/output_mixture.wav) and [silence](outputs/real/output_silence.wav)

## Original results on these recordings

| Calibration | Median F0 (Hz) | 5th–95th percentiles (Hz) | Voiced frames | Median normalized autocorrelation |
| --- | ---: | ---: | ---: | ---: |
| Female | 220.9 | 162.6–301.1 | 1,116 | 0.865 |
| Male | 107.7 | 95.3–181.7 | 3,386 | 0.946 |

All eight PCM32 mixtures match their saved reference sums with zero sample error. The largest measured ratio error was about 3.4e-11 dB. The five ratio mixtures peak at about 0.8913 full scale. Both methods were scored with the same fixed calibration. The largest reconstruction error when adding the saved float32 outputs was 3.7e-8 full scale.

| Female-to-male RMS (dB) | Harmonic F SI-SDRi (dB) | Harmonic M SI-SDRi (dB) | LPC F SI-SDRi (dB) | LPC M SI-SDRi (dB) |
| ---: | ---: | ---: | ---: | ---: |
| −10 | +2.68 | −1.44 | +2.69 | −1.55 |
| −5 | +2.30 | +0.21 | +2.33 | +0.17 |
| 0 | +1.52 | +1.44 | +1.54 | +1.44 |
| +5 | +0.56 | +2.17 | +0.54 | +2.23 |
| +10 | −1.20 | +2.39 | −1.28 | +2.48 |

At 0 dB the harmonic outputs achieved 1.57 dB female and 1.48 dB male SI-SDR. Source-correlated target gain was −3.88 and −4.78 dB, and interferer attenuation was 7.47 and 8.87 dB, respectively. At the two extremes, the already louder speaker lost SI-SDR after separation and the quieter output remained interference dominated. The LPC change was within 0.11 dB for every ratio and target, with mixed signs and effectively unchanged mean. The original release used the harmonic method by default.

The 0 dB paired excerpt contained 17.60 s of simultaneous activity, including 4.34 s classed as both voiced and 12.39 s of high-ZCR unvoiced overlap. The tracker came within 10% of isolated F0 on 46.9% of the female and 40.6% of the male reference-voiced frames. Female/male harmonic SI-SDR improvements were +1.81/+0.85 dB for both-voiced frames and +0.73/+0.64 dB in the high-ZCR subset. The harmonically related subset had 1.15 s and +1.53/−0.39 dB for the two targets. The close-F0 subset had only 0.07 s and was too short for its own score. With a two-second shift, overlap fell to 15.39 s and improvement rose to +2.47 dB for each voice. The isolated-voice runs still leaked into the empty channel at −9.64 dB RMS for female-only and −13.49 dB RMS for male-only, relative to the input. The male-only run also produced some spurious second-pitch detections.

The 67.03 s physical capture produced estimates with median tracked F0s of 207.5 and 112.5 Hz. The tracker reported two pitches on 0.88% of its active frames, compared with 3.45% in the controlled 0 dB run and 3.98% in the supplied clipped digital file. The real mask and spectrogram plots show shared broadband energy and intervals dominated by a single track. For example, around 15–23 s the male track is prominent and some low-frequency harmonic energy remains in the female estimate. The physical capture has no aligned reference pair, so its speaker leakage and SI-SDR remain unmeasured. The outputs were inspected numerically and through spectrograms. Audible artifacts were not assessed by listening.

On this Windows run the harmonic method processed the real recording in about 3.97 s, roughly 0.59 ms per frame and a 0.059 real-time factor. LPC used about 4.40 s and a 0.066 real-time factor. Eleven automated tests passed, including reconstruction with absolute tolerance `1e-12`, known synthetic pitches, exact WAV sums, and calibration partition isolation. CLI checks verified selection and profile reuse on a new mixture. The offline throughput leaves compute headroom for a streaming version, whose buffering and latency still need measurement.

## Toward live streaming

Use a mono callback and a bounded ring buffer, and move DSP off the audio callback. Carry the analysis window, pitch-state costs, mask smoothing and overlap-add state between blocks. Replace full-recording backtracking with a fixed-lag decoder and use a causal or bounded-lookahead smoother. Estimate the activity threshold online and measure end-to-end latency, underruns, peak CPU load and behavior on speaker changes. The current centered window alone requires 20 ms of future signal, and offline pitch backtracking uses the entire sequence.


## Repository release

This is version `v0.1.0-offline`, a pre-calibrated offline prototype. It does not support live microphone streaming. The reported 0.089 real-time factor is an offline throughput measurement, not a real-time capability. See `docs/optimization_report.md` and `docs/male_intelligibility_report.md` for experimental methodology, results and limitations. Aggregate tables and selected diagnostic figures are in `results/`. Human listening scores have not been collected.

### Planned work

Validate male output choices through listening and synchronized references from another session, then evaluate additional speakers and rooms. Develop bounded-lookahead tracking and causal smoothing before implementing live capture and mute controls.

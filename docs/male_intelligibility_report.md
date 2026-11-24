# Why the male output loses speech detail

**The baseline preserves male bass much better than male midrange.** In the balanced strict test, the reconstructed male contribution lost 0.62 dB at 80–300 Hz, 7.39 dB at 1–2.5 kHz and 9.63 dB at 2.5–4 kHz. Above 4 kHz, leaked female energy exceeded retained male energy by 18.69 dB. Raising the treble of the entire output would therefore amplify substantial interference. The physical recording shows a similarly bass-heavy male estimate, but its source-specific attenuation and intelligibility cannot be measured from that mixture alone.

**Two candidates merit listening first.** The 2 ms cepstral blend improved mean male SI-SDRi and spectral-envelope shape on development, validation and strict test. The 64 ms male analysis produced the highest strict-test SI-SDRi and preserved considerably more midrange. Development results for the longer window were mixed. The original 40 ms method remains the default. [Listening guide and audio](recommendation.md)

## A. Frequency regions and speech components

The baseline was rerun from the production inference path before editing it. All five ratios produced the archived strict-test float32 speaker samples exactly, including the 0 dB female/male scores of +7.08/+7.03 dB. All 15 original tests passed. The baseline archive preserves the previous source and outputs. The final digest comparison matched all 652 historical output files and all four source recordings. [Reproduction scores](diagnosis/baseline_reproduction.csv) · [Baseline backup](baseline.zip)

For each controlled mixture, infer the mask from the mixture and fixed profiles. For diagnosis, multiply each isolated source STFT by that same mask. Linearity permits an exact decomposition of the estimated male waveform after overlap-add. Reference waveforms, phases and voicing labels are confined to these measurements. The table uses the 0 dB strict mixture. The first two columns measure power change for each reconstructed component after reanalysis. The last column is the mean male mask weighted by male power in bins where male reference power exceeds female power by at least 9:1.

| Frequency | Male power passed, dB | Female power passed, dB | Female / male component power in output, dB | Mask on male-dominant bins |
|---|---:|---:|---:|---:|
| 80–300 Hz | −0.62 | −3.11 | −13.34 | 0.934 |
| 300–1,000 Hz | −5.90 | −12.77 | −1.72 | 0.416 |
| 1–2.5 kHz | −7.39 | −11.55 | +10.66 | 0.362 |
| 2.5–4 kHz | −9.63 | −9.73 | +9.14 | 0.273 |
| 4–8 kHz | −8.02 | −6.67 | +18.69 | 0.288 |

A direct comparison of output and reference magnitudes can conceal this loss. Above 4 kHz on male-active frames, mean output magnitude was 2.80 times the male reference and total band power was 9.96 dB higher. The true male component in the same frequency range suffered approximately 8 dB of attenuation over the recording. The retained-power measure in the variant sweep is computed on the masked analysis coefficients before overlap-add. Its values agree closely with the separately reconstructed baseline measurements above. [Rendered components](diagnosis/rendered_components.csv) · [Magnitude ratios](diagnosis/observed_spectra.csv) · [Reference, mixture and output spectra](diagnosis/baseline_spectra.png)

Both resonances and unvoiced detail were affected. On isolated-male voiced frames, LPC peaks ranked by frequency had median locations of 231, 2,002 and 2,776 Hz. In ±100 Hz regions around those peaks, reference-power-weighted male losses were 0.75, 6.95 and 8.75 dB. These are LPC resonance proxies. The lowest peak can follow excitation or spectral tilt. In the 2.5–4 kHz band, male component loss was 8.8 dB on voiced frames and 13.5 dB on active, high-ZCR frames lacking reliable male F0. The latter are a consonant-rich proxy rather than phonetic annotations. [Frame-level LPC resonances](diagnosis/male_lpc_resonances.csv) · [Voicing, band and mask measurements](diagnosis/components.csv)

In the physical recording, 83.8% of the original optimized male output’s FFT power lay at 80–300 Hz, versus 65.0% for the mixture and 23.6% for the female output. The male output carried 1.2% above 4 kHz. At 1–2.5 kHz its power was 6.92 dB below the mixture. This documents the output’s spectral balance. Speaker level, microphone response and leakage also affect that balance. [Physical recording measurements](diagnosis/real_output_distribution.csv)

## B. Which mechanisms explain the imbalance?

**The fitted spectral-power allocation is the main measured source of male midband loss.** With the 0 dB mixture and tracker fixed, the power-only mask already discarded 8.6 dB at 1–2.5 kHz and 10.3 dB at 2.5–4 kHz. Restoring the normal harmonic and LPC processing reduced those losses to 7.3 and 9.5 dB. At 2.5–4 kHz, removing final time and frequency smoothing recovered only 0.3 dB. The before-LPC and baseline losses there both rounded to 9.5 dB. The existing example-pair fit can assign more energy to the female channel even in bins independently known to be male dominant. These comparisons localize the error to the envelope/power allocation. They do not uniquely identify the responsible calibration exemplar or distinguish microphone characteristics from speaker identity.

**Mixture phase imposes a measurable secondary limitation.** Holding the estimated masked magnitude fixed and substituting the reference male phase moved male SI-SDRi from 7.03 to 9.12 dB. Holding mixture phase fixed and substituting reference magnitude yielded 15.15 dB. With reference magnitude and phase, the score reached the evaluation cap. These are diagnostic interventions on the same aligned mixture. The production candidates all use mixture phase. [Phase controls](diagnosis/phase_controls.csv)

**Spectral-envelope error is measurable but depends on what is compared.** On voiced male frames, the median observed-minus-reference 2 ms log envelope was −0.36 dB at 80–300 Hz and −2.20 dB at 2.5–4 kHz. Band RMSEs were 3.00 and 13.93 dB. Leakage and time variation can make an observed envelope look close in average level while its male component is attenuated. For the variant comparisons, shape RMSE removes each frame’s mean log-level difference over 80–8,000 Hz, then takes the RMS over bins and male-active frames. It measures the rendered estimate against the male reference and is sensitive to residual female energy. [Envelope measurements](diagnosis/observed_envelope_difference.csv)

## C. Variants and selection protocol

Development, validation and strict test used the existing PCM32 mixtures, sample ranges, guard intervals, ratios and calibration fractions. Development used the 40% calibration partition. Validation and test used 60%. All methods within a comparison received identical inputs. The strict test is the original 19.1095 s paired excerpt at five levels. Baseline diagnosis used that excerpt as requested. Candidate comparisons and settings were developed on the separate development/validation data and fixed before the strict candidate sweep. The five ratio rows share speech content. [Selection record](comparisons/selection.md)

| Variant | Male-path change |
|---|---|
| A current | Original conservative spectral-power mask, harmonic exponent 1.7, calibrated LPC and 40 ms window. |
| B power only | Fitted power envelopes and their uncertainty floor, with harmonic modulation and LPC removed. |
| C reduced harmonic | Halve pitch confidence in the modulation and use exponent 0.85. For a Gaussian comb this widens it by √2. Additional runs isolate each change. |
| D reduced smoothing | Reduce final Gaussian time/frequency smoothing from 0.85 frames / 0.55 bins to 0.25 / 0.15. Separate controls change one axis or reduce the fitted-band smoother from 1.25 to 0.40 frames. |
| E conservative mask | Relax the male mask toward one by up to 0.22 of the rejected share. Weight the lift by low-band male evidence and a frequency ramp from 300 to 1,800 Hz. |
| F cepstral assisted | Blend the baseline mask with a fitted cepstral-envelope power ratio, with at most 35% weight scaled by fit quality. |
| G power ratio | Normalize the baseline’s effective positive source powers raised to 0.75, with 0.55 as a second control. These quantities are already in power units. |
| H presence gain | Use the same male-evidence gate as E. Add at most 0.65 times the smaller of the passed and rejected shares in the presence band. This bounds the boost and keeps deeply rejected bins quiet. |
| W32, W64 | Recompute the male pipeline and calibration features using 32 ms or 64 ms Hann windows. Retain the 40 ms female waveform. All use a 10 ms hop, 1,024-point synthesis FFT, 16 kHz sample rate and centered overlap-add. |

The protected female waveform was checked sample-for-sample during the sweep. A is complementary. The optional variants apply an independent male mask or a second male analysis window, so the two outputs need not sum to the input. Each row in the metrics CSV records the maximum mixture-sum deviation. All masks are bounded. Output length, time grid and mute-female routing remain consistent.

## D. Strict held-out comparison

Values are arithmetic means across −10, −5, 0, +5 and +10 dB female-to-male RMS. Higher SI-SDRi, higher interferer attenuation and less-negative retained male power are desirable. Lower envelope RMSE is desirable. Interferer attenuation uses the original time-domain joint regression on both references, while the band columns measure passed source power.

| Male variant | SI-SDRi, dB | Male target gain, dB | Female attenuation, dB | Shape RMSE, dB | Male power at 2.5–4 kHz, dB | Male power at 4–8 kHz, dB |
|---|---:|---:|---:|---:|---:|---:|
| A current | 5.82 | −1.15 | 12.60 | 10.65 | −9.15 | −8.15 |
| B power only | 5.38 | −1.28 | 12.33 | 10.78 | −10.10 | −8.20 |
| C reduced harmonic | 5.75 | −1.19 | 12.79 | 10.69 | −9.69 | −8.20 |
| D reduced smoothing | 5.84 | −1.11 | 12.99 | 10.70 | −8.82 | −7.97 |
| E conservative mask | 5.35 | −1.12 | 10.08 | 10.60 | −7.42 | −7.08 |
| F cepstral assisted | **6.14** | −1.11 | 12.36 | **10.29** | −8.48 | −7.68 |
| G power ratio, 0.75 | 5.72 | −1.51 | 11.44 | 10.38 | −8.85 | −8.10 |
| G power ratio, 0.55 | 5.36 | −2.03 | 10.21 | 10.37 | −8.40 | −7.90 |
| H presence gain | 5.36 | −1.12 | 11.18 | 10.60 | **−6.97** | −6.69 |
| W32 | 4.62 | −1.76 | 11.54 | 10.47 | −7.99 | −5.20 |
| W64 | **6.74** | **−0.93** | **13.01** | 10.72 | −6.98 | −6.35 |

The isolated C confidence and width controls averaged 5.81 and 5.83 dB SI-SDRi. Reducing only final time smoothing, only frequency smoothing or only fitted-band smoothing gave 5.79, 5.87 and 5.54 dB. G moved ambiguous shares toward one half, which can also reduce the gain of already male-dominant bins. Its softer setting traded additional leakage for little envelope improvement. [All 240 split, ratio and variant scores](metrics.csv) · [Experiment averages](experiment_results.csv) · [Bandwise comparison plot](comparisons/preservation_and_leakage.png)

| Female-to-male RMS | Female, all variants | Male A | Male D | Male F | Male H | Male W64 |
|---:|---:|---:|---:|---:|---:|---:|
| −10 dB | +9.77 | +1.82 | +1.80 | +2.61 | +1.87 | +3.48 |
| −5 dB | +9.00 | +5.09 | +5.08 | +5.65 | +4.75 | +6.15 |
| 0 dB | +7.08 | +7.03 | +7.04 | +7.27 | +6.46 | +7.70 |
| +5 dB | +4.00 | +7.90 | +7.96 | +7.86 | +7.18 | +8.36 |
| +10 dB | −0.86 | +7.25 | +7.31 | +7.29 | +6.54 | +8.03 |

## E. What the cepstrum added

The real cepstrum uses the inverse real FFT of log magnitude plus a numerical floor inside the logarithm. Low-quefrency liftering retains coefficients within 1 or 2 ms of index zero **in both circular directions**. Transforming that even sequence back yields the smoothed log-magnitude envelope. At 16 kHz these cutoffs retain up to 16 and 32 samples on each side. No step subtracts isolated or mixture cepstra to obtain a source. [Example envelopes from voiced and unvoiced frames](diagnosis/cepstral_envelopes.png)

A nearest-example diagnostic on isolated development speech found almost no room to improve speaker classification. Ordinary band powers correctly identified 99.9% of voiced male frames and all 116 active unvoiced male frames. The 2 ms envelope matched 99.8% and 100%, while 1 ms matched 98.6% and 93.1%. The reason to test F came from the **mixture** probe. At development 0 dB, A, B, the 1 ms blend and the 2 ms blend gave male SI-SDRi of 8.13, 7.50, 8.08 and 8.37 dB. The 2 ms blend also reduced shape RMSE from 11.45 to 11.02 dB and increased 2.5–4 kHz retained power by 0.36 dB. [Isolated matching](diagnosis/cepstral_matching.csv) · [Mixture probe](diagnosis/cepstral_development_probe.csv)

F uses up to 16 active voiced and 16 active unvoiced calibration examples per speaker. Exponentiating twice the liftered log magnitude yields an envelope in power units. The existing nonnegative two-example fit operates on these band powers, estimates one gain per speaker, and supplies a fit-quality weight. The mixture-derived power ratio is interpolated onto the FFT bins and blended into the baseline male mask. Calibration sources and sample ranges are recorded in the profile. Existing profiles still load, and F explicitly requires the optional cepstral bank.

Relative to A, F improved mean male SI-SDRi by 0.24 dB on development, 0.18 dB on validation and 0.32 dB on strict test. Its strict-test shape RMSE improved by 0.36 dB and its broadband female attenuation declined by 0.24 dB. The gain is modest but repeatable across the partitions used here.

## F. Leakage and the longer window

W64 improved strict-test male SI-SDRi across all five ratios and improved overall female attenuation by 0.41 dB. At strict 0 dB it recovered 2.26 dB of 300–1,000 Hz male power and 2.57 dB at 2.5–4 kHz. At 4–8 kHz it also passed 1.16 dB more female power. On development, W64 averaged 6.66 dB SI-SDRi against 7.35 dB for A. On validation the figures were 7.00 versus 6.81 dB. This sensitivity argues for a listening comparison and independent recordings before switching defaults.

The 64 ms window changes the physical observation duration and the calibrated spectra. All windows have the same synthesis bin spacing of 15.625 Hz. Zero-padding samples the spectrum more densely without extending the observed signal. At strict 0 dB the 32/40/64 ms trackers got both pitches correct on 46.4/49.0/44.6% of 345 reference-both-voiced frames. The W64 separation gain therefore has no corresponding increase in correct pitch pairs. This end-to-end comparison cannot disentangle window length from the calibration features it changes.

For 71 isolated-male high-band attacks, the reference median rise over 20 ms was 12.30 dB. The 32/40/64 ms estimates’ retained male components rose by 10.29/12.86/12.89 dB. Median absolute rise errors were 1.80/2.71/2.50 dB. These events came from a common 10 ms physical-time grid and a 2.5 kHz high-pass, with a 6 dB rise threshold. The 64 ms result gives no evidence of extra attack loss on this specific test. Its 32 ms centered-window lookahead exceeds the baseline’s 20 ms. [Window and attack diagnostics](diagnosis/window_attacks_and_pitch.csv)

## G. Runtime, validation and listening

On the full 67.033 s physical recording, the measured processing times and real-time factors were A 5.66 s / 0.085, D 5.95 s / 0.089, F 6.42 s / 0.096, H 5.78 s / 0.086 and W64 11.48 s / 0.171. The W64 time includes the protected 40 ms pass plus the separate 64 ms pass. Other variants include the shared analysis time and the work needed to return both outputs. These are offline timings from this machine, excluding calibration, scoring, disk I/O and loudness matching. [Real-recording timings](listening/real/runtime.csv)

All 22 automated tests passed. The tests cover lifter symmetry, full-envelope reconstruction, zero and extreme powers, frequency boundaries, silence, one-sample input, output length, aligned references, window reconstruction, female protection, mute routing and saved-baseline regression. Independent CLI renders for F and W64 matched the listening masters exactly on the controlled mixture.

Both listening sets contain the entire aligned recording, unity-gain masters under `raw/`, the original female output, mixture, ten male alternatives and a blank listening sheet. The controlled set also includes the isolated male reference. Each copy is matched to −23 LUFS using mono K weighting at 48 kHz, 400 ms blocks, 100 ms steps, an absolute −70 LUFS gate and a relative −10 LU gate. The files were measured again after writing. The largest level difference within either set was under 0.008 LU. Fourfold oversampling gave maximum peak amplitudes of 0.703 on the physical set and 0.457 on the controlled set. [Physical levels](listening/real/levels.csv) · [Controlled levels](listening/controlled_0dB/levels.csv) · [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770/en)

## H. Limits and reproduction

There is one recording per calibrated identity and one unreferenced shared-microphone capture. Clean-source power, formant proxies and cepstral distortion can identify missing cues, but neither those metrics nor SI-SDR establish word intelligibility. Residual female speech can improve an output’s apparent spectral balance. The reported listening observation came from the user. The listening sheets await human ratings.

To reproduce the study from the workspace root, use the existing optimization fixtures and fixed profiles. The study writes under `outputs/male_intelligibility`. The evaluation scripts use references for scoring, calibration-partition checks and explicit oracle diagnostics. The deployed separator takes mixture audio and calibrated profiles.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python male_study.py reproduce
python male_study.py diagnose
python male_study.py pilot
python male_study.py cepstrum
python male_study.py development
python male_study.py validation
python male_study.py test
python male_study.py windows
python male_study.py collect
python male_study.py listening
python -m unittest discover -s tests -v
```

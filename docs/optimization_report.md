# Classical two-speaker separation on the supplied recordings

The revised separator improved both speakers at every tested ratio relative to the original separator. Across ten speaker-by-ratio scores on the untouched 19.1095 s test excerpt, mean SI-SDR improvement rose from **1.07 to 5.81 dB**. In the equal-level mixture, correct simultaneous pitch pairs rose from **16 to 169 of 345** reference-voiced overlap frames. The largest separation gain came from estimating each speaker’s spectral power before forming the mask.

The louder female target still lost 0.86 dB relative to the unprocessed +10 dB mixture. Both single-speaker controls leaked more energy into the unused output. These failures matter for selective muting.

## A. Reproduction and root causes

Before changing the algorithm, I archived the working tree’s original source, tests, README, fixed profile and historical scores. The workspace had no Git repository. All 11 original tests passed. A fresh run of the original CLI reproduced all 208 historical score rows. The maximum absolute difference across reported dB fields was 1.75 × 10⁻¹² dB. On the historical 0 dB excerpt the original F0 accuracies were 46.90% female and 40.64% male. Dual detection was 3.45% of mixture-active frames and **22/434 = 5.07% of simultaneous reference-voiced frames**. The reproduction took 4.91 s for the 67.03 s physical recording, RTF 0.073. The historical timing was 0.059. [Baseline snapshot](baseline_snapshot.zip) · [Original CLI reproduction](reproduction/evaluation.json)

**The joint scorer discarded usable candidates.** On the development 0 dB mixture, the original shortlist contained both correct F0s within 10% on 255 of 303 both-voiced frames. Its instantaneous best state contained two pitches on 44 frames. The full-sequence path retained both on just one. The secondary harmonic score sums support over the residual peak mass, while the old speaker prior and voicing deduction were independent of that mass. This disproportionately charged a quiet second voice. The following controlled changes use the same development frames and reference estimator. [Candidate and transition experiments](pitch_diagnostics/development_tracker.csv)

| Development tracker, 0 dB | Both candidates available | Both pitches reported | Both identities and F0s correct |
|---|---:|---:|---:|
| Original | 84.2% | 0.3% | 0.3% |
| Lower transition cost alone | 84.2% | 4.3% | 1.0% |
| Residual-scaled costs alone | 92.7% | 37.0% | 27.4% |
| Both changes | 92.7% | 60.4% | 42.6% |
| Both changes plus refined spectral peaks | 90.8% | 59.7% | 44.9% |

Temporal penalties contributed independently. With the revised emissions, reducing transition costs increased correct pairs from 27.4% to 42.6%. Peak refinement improved the fraction of correct selections despite a small decrease in shortlist coverage. On the strict test at 8.66 s, the old path selected 217.5 Hz and an absent male pitch. The revised path selected 200 and 105 Hz, close to both isolated estimates. Other frames genuinely lacked a correct proposal. [Rejected pair](pitch_diagnostics/test_rejected_pair.png) · [Missing candidate](pitch_diagnostics/test_missing_candidate.png) · [Remaining error](pitch_diagnostics/test_remaining_failure.png)

**The old fallback split energy roughly in half when pitch supplied little evidence.** On the strict 0 dB test, bins containing at least 90% target reference power received an energy-weighted mean target mask of 0.640 female and 0.604 male. The revised default assigned 0.803 and 0.892. In frames where the respective tracker returned neither pitch, the old allocations averaged 0.503 and 0.509, compared with 0.723 and 0.811 using the revised mask. Those absent-pitch frame sets differ by tracker. The fixed-tracker ablation below isolates the mask effect. Both algorithms enforce complementarity, so this improvement is achievable while retaining mixture consistency. [Allocation measurements](mask_comparison/test_allocation.csv) · [Mask comparison against isolated-source power](mask_comparison/test_female_masks.png)

**The LPC model was stable but contributed little under the original mask.** Recomputing the saved centered log envelopes from their coefficients agreed within 1.1 × 10⁻¹³. All 48 filters had pole radii below 0.999. On the strict balanced test, 72.0% of old per-speaker/bin multipliers were exactly one. Their 5th and 95th percentiles were 0.916 and 1.157. When pitch confidence vanished, the adjustment could do nothing to the old background term. Median template cosine similarities were 0.083 within the female bank, 0.620 within the male bank and −0.082 across banks. Within-speaker envelope variability is measurable. Phonetic labels are unavailable, so attributing that variation to particular phonemes remains a hypothesis. [LPC and dropout diagnostics](mask_comparison/test_lpc.json)

The centered frame, Hz/bin, sample-lag, cepstral search and inverse-STFT conventions were consistent with the original synthetic-frequency and reconstruction checks. The direct normalized-correlation and frame-center comparisons are in the verification record. No observed indexing, scaling or overlap-add error explained the low historical scores. The calibrated priors, octave ambiguity and the isolated voicing gate still influence errors. The experiments establish the cost-scaling problem, but do not establish that the calibrated medians themselves were wrong. [Verification](verification.json)

## B. Changes and experimental design

The inference path still accepts a mono waveform and a fixed calibration profile. All changes use autocorrelation, cepstrum, harmonic analysis, classical dynamic programming, short-time spectra and matching against isolated examples.

1. **Pitch evidence.** The tracker samples spectral peaks with a 2048-point zero-padded transform of the same 640-sample frame. This halves the sampling interval of the spectrum while preserving the analysis window. The whitening floor drops from 6% to 2.5% of the frame maximum. The 70–350 Hz candidate grid, 2.5 Hz step, ACF/cepstrum, calibrated F0 priors and two orderings of residual scoring remain in use. Peak support already explained by the first comb is discounted nonnegatively.
2. **Joint selection.** Write the residual mass as *m* and the existing secondary periodicity, voicing and prior term as *b*. The old adjustment was *b − 0.16√m*. The revised adjustment is *m(b − 0.09) − 0.012*. The small fixed cost controls weak residuals. Multiplying jump and presence-change costs by 0.35 permits shorter, moving second trajectories. An explicit absent pitch remains an option for each identity.
3. **Spectral power.** Each speaker contributes up to 16 evenly spaced voiced and 16 active unvoiced examples from calibration. Each example is a normalized short-time band-power spectrum on geometrically spaced frequency bands. For a mixture band vector *y*, test each female/male example pair *a, b* and fit nonnegative gains *u, v* by minimizing the weighted squared error of *y − ua − vb*. Each band’s amplitude weight is the reciprocal of its mixture power plus 3% of the frame maximum, with a numerical floor. The solver checks the two-gain solution and the single-source boundaries, including collinear examples. This is fixed feature matching with two level estimates per frame.
4. **Soft mask.** Smooth fitted band powers in time with a 1.25-frame Gaussian, convert to power per bin and interpolate across frequency. For each voice, modulate this envelope with its Gaussian harmonic comb raised to 1.7, normalized to unit frequency mean and blended by pitch confidence. The conservative variant adds a share of the two-source mean envelope that grows as weighted fit quality decreases. The floor fraction is 0.02 + 0.25(1 − fit). Apply the optional LPC factor, normalize the independent positive strengths, smooth with the original 0.85-frame and 0.55-bin Gaussian, and use complementary masks with the original mixture phase. There is no unassigned residual.
5. **Default.** The full bounded LPC adjustment improved both speakers at all five development and validation ratios on the new conservative mask. Validation mean SI-SDRi rose from 0.81 dB for the original system to 6.90 dB for the selected system. The CLI default became `harmonic_lpc` before the final test. `harmonic` omits the LPC factor. Older profiles use the pitch-comb fallback until recalibrated. The original profile is preserved alongside the reproduced evaluation.

**Source separation between experiments.** There is one isolated recording per speaker. For development, calibration used the first 40% of each source. A 250 ms guard preceded the 12 s development excerpts. After choosing the tracker and shortlisting masks, calibration used the original first 60%, which includes the development audio. The final calibration excluded all validation and test samples. Validation used the next 6 s after the original guard. A second 250 ms guard preceded the untouched test. All methods within a comparison received identical PCM32 sums. Each excerpt was independently RMS-normalized at each ratio. The full historical benchmark includes the validation excerpt and is reported separately from the strict test.

| Source range, 16 kHz samples, half-open | Female | Male |
|---|---:|---:|
| Development calibration | [0, 409752) | [0, 441560) |
| Development mixture | [413752, 605752) | [445560, 637560) |
| Final calibration | [0, 614628) | [0, 662340) |
| Validation mixture | [618628, 714628) | [666340, 762340) |
| Strict final test | [718628, 1024380) | [766340, 1072092) |

## C. Held-out results

SI-SDRi is the original centered, projected SI-SDR of the saved float32 estimate minus the score of the mixture. All scores below are dB. Each ratio reuses the same excerpt, so the five rows represent level conditions on the same speech.

| Female-to-male RMS | Strict original F / M | Strict optimized F / M | Full historical original F / M | Full historical optimized F / M |
|---:|---:|---:|---:|---:|
| −10 dB | 2.67 / −1.25 | 9.77 / 1.82 | 2.68 / −1.44 | 10.14 / 2.08 |
| −5 dB | 2.26 / 0.29 | 9.00 / 5.09 | 2.30 / 0.21 | 9.34 / 5.35 |
| 0 dB | 1.59 / 1.51 | 7.08 / 7.03 | 1.52 / 1.44 | 7.34 / 7.24 |
| +5 dB | 0.56 / 2.18 | 4.00 / 7.90 | 0.56 / 2.17 | 4.15 / 8.03 |
| +10 dB | −1.30 / 2.19 | −0.86 / 7.25 | −1.20 / 2.40 | −0.75 / 7.35 |
| Mean over both targets and ratios | 1.07 | **5.81** | 1.06 | **6.03** |
| Worst individual score | −1.30 | **−0.86** | −1.44 | **−0.75** |

The weakest strict-test ratio by the mean of its two target scores was +10 dB in both systems, improving from 0.45 to 3.20 dB. The female minimum improved from −1.30 to −0.86 dB and the male minimum from −1.25 to 1.82 dB.

At strict-test 0 dB, female/male target attenuation fell from 3.88/4.68 to 2.23/1.05 dB. Interferer attenuation rose from 7.61/8.87 to 18.86/12.91 dB. These are joint regression coefficients, reported as signed target gain and positive interferer attenuation in the CSVs. The two-second staggered test improved from 2.86/2.85 to 7.52/7.36 dB. An absent target has undefined SI-SDRi. In the female-only control, RMS leakage into the male output worsened from −9.95 to −8.13 dB relative to input. In the male-only control, leakage into the female output worsened from −14.10 to −12.87 dB. [Baseline metrics](offline-results/baseline_metrics.csv) · [Optimized metrics](offline-results/optimized_metrics.csv)

On the strict balanced test, 69.5% of 1,912 frames had both references active, 18.0% had two reliable isolated pitches, 57.2% had one and 24.7% had neither. Each isolated signal uses the existing ACF peak and confidence, energy and ZCR gates. Reference and mixture frames share the same centered 10 ms grid. The independent estimates are proxies and can miss genuine voicing.

| Strict 0 dB pitch diagnostic | Original | Improved |
|---|---:|---:|
| Dual presence on mixture-active frames | 3.61% | 40.80% |
| Dual presence on 345 reference-both-voiced frames | 4.64% | 57.10% |
| Both assigned F0s within 10% on those frames | 4.64% | 48.99% |
| Within 10% on each speaker’s voiced frames, F / M | 40.88% / 41.81% | 61.07% / 69.77% |
| Within 10% conditional on both voiced, F / M | 33.91% / 42.90% | 56.81% / 80.29% |
| Missed voiced frames, F / M | 218 / 793 | 131 / 386 |
| Spurious voiced frames, F / M | 322 / 1 | 558 / 4 |
| Half-frequency errors, F / M | 0 / 6 | 2 / 28 |
| Double-frequency errors, F / M | 5 / 0 | 5 / 0 |
| Unambiguous swapped frames / adjacent identity switches | 0 / 0 | 0 / 0 |

A miss means zero output F0 on a voiced reference. A spurious detection means positive F0 on a reference lacking reliable pitch. An octave error is within 10% of half or twice the isolated F0. A swapped frame matches both crossed identities within 10% and fails the direct-pair test. A switch is a change between unambiguous direct and crossed assignments on consecutive reference-both-voiced frames. Gaps and near-equal ambiguous pairs cannot establish identity. One-frame pitch dropouts, counted as present/absent/present, increased from 2/0 to 6/4. On the full historical 0 dB mixture the revised dual-presence rate was 248/434 (57.14%) and the both-correct rate was 210/434 (48.39%). [All pitch diagnostics](pitch_diagnostics/held_out.csv)

## D. Component and oracle ablations

Each row uses the strict test’s five identical ratio mixtures. Oracle rows replace F0 and voicing with the isolated-reference estimates and set confidence to one on their voiced frames. Their timing excludes reference extraction. This diagnostic changes confidence as well as F0, and the isolated estimates themselves are gated. Its scores describe mask behavior under those assumptions. Reference waveforms and reference F0 only enter the offline experiment and scoring code. The deployed analyze/separate path has neither as an input.

| Tracker | Mask | Female mean SI-SDRi | Male mean SI-SDRi | Worst |
|---|---|---:|---:|---:|
| Original | Original | 1.16 | 0.99 | −1.30 |
| Improved | Original | 1.86 | 1.76 | −1.20 |
| Original | Selected power mask + LPC | 5.77 | 5.77 | −0.68 |
| Improved | Selected power mask + LPC | **5.80** | **5.82** | −0.86 |
| Oracle, evaluation only | Original | −0.16 | 1.32 | −7.18 |
| Oracle, evaluation only | Selected power mask + LPC | 6.17 | 6.12 | −0.87 |
| Improved | Independent powers, normalized | 5.31 | 5.59 | −1.68 |
| Improved | Conservative power mask, harmonic only | 5.61 | 5.62 | −1.00 |
| Improved | Conservative + half LPC log-weight | 5.71 | 5.72 | −0.93 |
| Improved | LPC-only shape likelihood | 0.50 | 0.48 | 0.10 |

With original masks, better tracking gained 0.74 dB on average. With the original tracker, the selected mask gained 4.70 dB. With the selected mask, changing tracker added only 0.04 dB on average and cost 0.18 dB in the worst case, despite the clear improvement in pitch accuracy. The isolated-F0 control added about 0.34 dB over deployed mean performance with the selected mask. Oracle F0 did poorly with the original mask, which still divided unpitched energy and applied full harmonic confidence on every reference-voiced frame. Better F0 alone cannot resolve that allocation problem.

The LPC-only control used the original matched LPC envelopes at full confidence with harmonic and band-power evidence removed. The half-weight control applied the square root of the full LPC multiplier. All non-oracle LPC rows shared identical mixture-derived tracks. Development sensitivity checks with LPC disabled gave female/male means of 7.22/7.12 dB with the chosen comb, 6.81/6.70 with no harmonic modulation, 7.24/7.14 with a wider comb and 7.12/7.03 with a narrower comb. Removing final time smoothing gave 7.12/7.09. Removing final frequency smoothing gave 7.32/7.21. The smaller smoothing and width effects leave room for further validation. [Full ablations](offline-results/ablation_results.csv) · [Development sensitivity](mask_comparison/development_sensitivity.csv) · [Validation](mask_comparison/validation.csv)

## E. Limits and runtime

On the full historical 0 dB mixture, harmonically related overlap improved from 1.53/−0.39 to 6.00/4.77 dB. The distinct-F0 subset reached 8.69/7.64 dB. Close-F0 overlap lasted only 0.07 s, below the original 0.25 s reporting minimum. High-ZCR unvoiced overlap improved from 0.73/0.64 to 7.46/7.36 dB. Yet above 4 kHz in that strict-test overlap subset, the energy-weighted male mask share fell from 0.487 to 0.308. The calibrated broadband envelopes can favor one speaker’s consonants too strongly. Shared harmonics, similar source spectra and mixture phase also leave unresolved energy. [Full subset scores](final_cli/metrics.csv)

The unreferenced 67.03 s physical recording yielded two reported pitches on 13.72% of active frames, up from 0.88%. There are no aligned physical references or listening-test ratings. One recording per identity means the study cannot distinguish speaker identity from persistent recording-channel characteristics or establish performance on another session. The supplied clipped digital sum includes calibration content and contributes no held-out quality score.

Three interleaved single-threaded-BLAS measurements on the physical recording gave median processing times of **4.27 s original and 5.98 s optimized**, RTF **0.064 and 0.089**. These measure offline STFT, tracking, masks and reconstruction, excluding file I/O and plotting. The full CLI, including experiments run under different machine load, records its own timings. The centered window uses 20 ms lookahead, the activity floor sees the full recording, and full-sequence backtracking and symmetric smoothing use future frames. Throughput alone cannot establish live latency. [Timing repetitions](real_timing.csv)

## F. Recommendation and reproducibility

**Separation quality remains the blocker for dependable live muting.** The system is fast enough for a fixed-lag prototype. First validate the louder-female and single-speaker failures with independently recorded synchronized source and room-microphone signals, and assess consonant loss by listening. A streaming study can then retain the same state emissions, carry the previous DP costs across blocks, backtrack within a bounded delay, replace the global activity threshold and bound smoothing and overlap-add lookahead.

All 15 tests passed, including the weak moving pitch, calibrated unpitched allocation, collinear/silent power fit, known dual-pitch identity, mute routing, exact PCM sums and sample-edge reconstruction. All eight full-benchmark PCM triplets matched the original input files byte-for-byte. Independent optimized CLI and experiment renders produced identical float32 speaker samples on all eight. The maximum float64 reconstruction error across the ablations was 4.45 × 10⁻¹⁶ full scale. Saved float32 outputs reconstructed the mixture within 4.29 × 10⁻⁸. CLI inspection, calibration, separation, visualization and evaluation ran successfully. Each of the four CLI mute modes produced the expected selected samples. [Verification record](verification.json) · [Strict test female WAV](test/ratio_+0dB/output_female.wav) · [Strict test male WAV](test/ratio_+0dB/output_male.wav) · [Physical recording estimates](final_cli/real/harmonic_lpc/)

Run from the workspace root with the supplied WAVs. The measured environment used Python 3.14, NumPy 2.5.1, SciPy 1.18.1 and Matplotlib 3.11.2. Run the historical CLI from the archived source to reproduce the original directory. Its `evaluate` command accepts explicit source, recording, supplied-digital and profile paths. The following commands regenerate the optimized studies and checks in their dedicated directories.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m unittest discover -s tests -v
python optimization.py tracker
python optimization.py development
python optimization.py validation
python optimization.py test
python optimization.py held_out
python optimization_diagnostics.py development --mask-sensitivity
python optimization_diagnostics.py test
python optimization_diagnostics.py held_out
python main.py calibrate --out-dir outputs/optimization/final_cli
python main.py evaluate --out-dir outputs/optimization/final_cli
python main.py separate --input outputs/optimization/test/ratio_+0dB/Digi_added_samples.wav --mode mute-male --profiles outputs/optimization/final_cli/profiles.json --out-dir outputs/optimization/test/cli_mute_male
python optimization.py verify
python optimization.py benchmark
```

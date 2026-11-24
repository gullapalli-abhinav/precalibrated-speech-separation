# EC4143 real-time DSP architecture

The application estimates two enrolled speakers from a single microphone. Its output router selects one estimate according to the debate controller. The separation engine processes both speakers throughout the session. This document describes the streaming path used by the desktop app.

## Signal path and active techniques

| Stage | Active streaming behavior | Source |
|---|---|---|
| Input and enrollment | Full-duplex mono float32 capture. A causal rational polyphase FIR converts hardware audio to 16 kHz when needed. Separate enrollment WAVs may be imported. Each speaker's profile is fixed during playback. | [LiveAudio and Converter](../src/live_audio.py), [enroll and read_session](../src/enrollment.py), [WAV import](../src/audio_io.py) |
| Framing and pitch | Periodic Hann STFT, frame energy gate, ZCR penalty, overlap-normalized STACF, real-cepstrum peaks, refined harmonic peak evidence, profile priors and a joint two-hop fixed-lag decoder. | [StreamingSeparator._frame and FixedLag](../src/streaming.py), [pitch_candidates](../src/pitch_tracking.py) |
| Spectral power | Weighted nonnegative fit to two fixed banks of band-power exemplars. Smooth the fitted powers across a small frame neighborhood, then interpolate power density across frequency. | [match_band_powers](../src/separator.py), [StreamingSeparator._frame](../src/streaming.py) |
| Mask refinement | Harmonic confidence modulation and LPC shape matching feed a smoothed female power ratio. A 2 ms cepstral correction can modify the male complement if both profiles contain the special bank. Cepstral evidence in pitch tracking is active independently. | [StreamingSeparator._mask](../src/streaming.py), [log_envelope](../src/cepstrum.py) |
| Synthesis and selection | Mask the complex mixture STFT, then inverse FFT and weighted overlap-add. Capture-time turn decisions route a speaker estimate, mixture, or silence. Speaker switches use a configurable crossfade. Playback gain acts on the device signal after routing, while the selected recording uses the routed signal before playback gain. | [StreamingSeparator](../src/streaming.py), [Router](../src/debate_controller.py), [LiveAudio._emit](../src/live_audio.py) |

**Offline-only or diagnostic paths.** The original CLI uses full-sequence Viterbi backtracking, a whole-recording relative energy gate, batch symmetric temporal smoothing and full-array inverse STFT. Its legacy pitch-comb fallback and experimental male variants, including the optional 64 ms alternate pass, are outside the active live path. The streaming constructor's bypass option is for reconstruction checks. Enrollment LPC pole frequencies and bandwidths produce formant diagnostics, while the streaming mask uses the saved log envelopes. Harmonic amplitude medians and pitch histograms are recorded as calibration summaries. They are not added to the live mask. The live separator reads the profile pitch median/spread, P95 energy, band-power banks, LPC log envelopes and optional cepstral banks.

## Calibration and analysis parameters

| Parameter | Live value |
|---|---|
| Internal rate and framing | Mono 16,000 Hz, periodic Hann, 640 samples (40 ms), hop 160 (10 ms), 1,024-point real FFT with 513 bins at 15.625 Hz spacing |
| Joint pitch search | 70–350 Hz at 2.5 Hz spacing. Peak evidence uses a 2,048-point FFT and 55–3,500 Hz band |
| Voicing and lag | Enrollment correlation cutoff 0.52 and ZCR below 0.18. Live ZCR penalty starts at 0.18. Live traceback lag 2 hops (20 ms) |
| Power exemplar maximum | Up to 16 voiced and 16 unvoiced frames from active speech per speaker and per bank |
| LPC | Order 14. Up to 24 saved envelopes per profile. Match up to three per frame over 200–4,500 Hz |
| Power and mask smoothing | Band powers use up to two earlier and two later frames with Gaussian sigma 1.25 hops. Female mask frequency sigma 0.55 bins and temporal weights 0.72 current, 0.28 previous |
| Cepstral correction and crossfade | Optional symmetric lifter cutoff of 2 ms (32 samples), blend strength at most 0.35. Default route crossfade 20 ms |
| Capture horizon | 640 + 2 × 160 = 960 samples (60 ms) from the start of an emitted hop, before hardware and queues |

### Enrollment

[Enrollment](../src/enrollment.py) checks finite mono input, usable duration, excessive clipping, adequate frame energy and sufficient reliable pitch. The GUI captures 30, 45, 60 or 90 seconds or imports isolated recordings. Live captures are not retained as raw calibration WAVs. [calibrate_speaker](../src/calibration.py) uses centered frames and measures energy, ZCR, pitch and cepstral agreement. It stores a median F0, voiced percentiles, log2 pitch scale, active-frame energy percentiles and example banks. Each spectral power exemplar is normalized by its total band power. New live profiles contain both the usual band powers and the optional cepstral powers.

For a mean-removed acquisition frame $u[n]$, the normalized short-time autocorrelation is

$$
r[\ell]=\frac{\sum_{n=0}^{639-\ell}u[n]u[n+\ell]}
{\max(\epsilon,\sqrt{\sum_{n=0}^{639-\ell}u[n]^2\sum_{n=0}^{639-\ell}u[n+\ell]^2})}.
$$

The implementation evaluates the numerator by a zero-padded FFT and the denominator by prefix energy sums. It uses lags 46–228 samples for isolated pitch peak selection. The detector picks the earliest local maximum within 0.13 of the strongest peak. A three-point parabola refines the chosen lag by at most half a sample. Then $T_0=(\ell+\delta)/F_s$ and $F_0=1/T_0$, with the frequency clipped to the pitch search range. A cepstral lag estimate is also recorded when cepstral evidence passes its cutoff. Peaks can occur at multiples of a true period, while dominant harmonics can favor a shorter period. Unvoiced speech does not offer a stable periodicity.

### Streaming features and joint pitch

For frame center $tH$, with startup zero padding and $n=0,\ldots,639$,

$$
w[n]=0.5-0.5\cos(2\pi n/640),\qquad
X_t[k]=\sum_n x[tH-320+n]w[n]\,e^{-j2\pi kn/1024}.
$$

The unwindowed frame gives $E_t=\sum_n x_t[n]^2/640$ and ZCR as the fraction of adjacent sign changes. The live pitch gate compares $E_t$ with $\max(10^{-10},10^{(\min_s P95_s-42)/10})$, where $P95_s$ is that speaker's calibrated active-frame energy in dBFS. The real cepstrum is the inverse FFT of log magnitude. The pitch path floors magnitude relative to the frame peak, subtracts a Gaussian-smoothed cepstral baseline (sigma 4), and normalizes/clips the positive prominence. It linearly interpolates STACF and cepstral prominence at $F_s/f$.

For candidate pitch $f$, the primary emission combines coverage-corrected harmonic peak score $h(f)$, periodic support and speaker prior:

$$
e_s(f)=h(f)+0.10\max(r[F_s/f],0)+0.07c[F_s/f]-0.31
-0.24\,\mathrm{clip}((\mathrm{ZCR}-0.18)/0.15,0,1)
-\min(0.055d_s^2,0.65),
$$

where $d_s=|\log_2(f/\mathrm{median}_s)|/\max(0.13,\mathrm{scale}_s)$. Peak scoring whitens spectra and discounts peaks explained by the other candidate speaker. Each speaker has three primary proposals, three residual proposals and an unvoiced option, giving 49 joint states. The two-voice emission takes the better primary-plus-residual ordering and penalizes nearly identical pitches. See [pitch_candidates](../src/pitch_tracking.py) for the residual mass and Gaussian coincidence terms.

With pitch change $\Delta=|\log_2(f_t/f_{t-1})|$, each speaker costs $\min(1.1\Delta+5\Delta^2,2)$ when voiced on both frames, 0.14 when toggling voicing, and zero when both frames are unvoiced. The recurrence is $V_t(i)=e_t(i)+\max_j[V_{t-1}(j)-0.35\,\mathrm{cost}(j,i)]$ with a sum over speaker costs. Live traceback commits the state two hops back. Confidence clips $(e_{\rm chosen}-e_{\rm without\ speaker}+0.03)/0.34$ to [0,1] for voiced states. This score is heuristic. Close pitches and overlapping periodicities can confuse the speaker estimates.

### Spectral power, LPC and masks

The band fit sums $P_b=\sum_{k\in b}|X_t[k]|^2$. Band starts come from rounded geometric spacing across 75–8,000 Hz plus the DC band, with duplicate starts removed. For every pair of normalized templates $a_f,a_m$, the code fits gains by minimizing

$$
\left\|W(P-g_fa_f-g_ma_m)\right\|^2,\qquad
W_{bb}=1/\max(P_b+0.03\max_b P_b,\epsilon),\qquad g_f,g_m\ge0.
$$

The implementation tests the interior solution or the best single-source boundary per pair, then chooses the greatest explained weighted power. Its quality score $q$ is explained power divided by observed weighted power, clipped to [0,1]. Fitted bands are smoothed over available offsets using weights $\exp(-d^2/(2\cdot1.25^2))$. Dividing by band widths and interpolating yields power density $D_s[k]$. The uncertainty term adds $(0.02+0.25(1-q))$ times the mean speaker density at each bin.

The synthesis comb is centered on the nearest positive harmonic at width $\min(1.10F_s/1024+0.009f_k,0.24F_0)$ and raised to 1.7. For confidence $C_s$ and comb $G_s$, the expression $1+C_s(G_s/\max(\mathrm{mean}(G_s),0.05)-1)$ redistributes the fitted density. When no pitch is tracked, this multiplier is one.

LPC fits a 14th-order predictor to pre-emphasized, windowed enrollment frames. It solves the autocorrelation Toeplitz system $Ra=-r$ with diagonal regularization. The all-pole log magnitude $-\log|\sum_{i=0}^{14}a_i e^{-j2\pi ki/1024}|$, with $a_0=1$, is centered and stored. The live engine compares the mean-centered, unit-norm mixture log shape with these saved shapes after smoothing the magnitude with a Gaussian sigma of six bins and adjusting its slope for pre-emphasis. The top three matching log envelopes give a frequency-dependent multiplier $\exp(0.22 C_s\,\mathrm{clip}(L_s-\mathrm{median}(L_s),-2,2))$. No live LPC coefficient fitting is required.

From the resulting strengths $S_f,S_m$, $M_f$ starts at $S_f/(S_f+S_m)$, then receives frequency smoothing and a causal temporal update. The male base is $1-M_f$. If both optional banks are present, retain cepstrum samples with circular quefrency $|n|\le 32$ and transform back to obtain log envelope $L$. Run the same band fit on $\exp(2L)$, and interpolate the male ratio $R_m$ from those fitted bands. The update is

$$
M_m=(1-M_f)+0.35q_{\rm cep}\left[R_m-(1-M_f)\right].
$$

This correction can improve broad envelope assignment without relying on exact harmonic locations. Only the optional correction breaks complementarity. For each speaker the actual applied complex spectrum is $Y_s[t,k]=M_s[t,k]X_t[k]$, with mixture phase preserved.

### Inverse STFT and route

The engine takes the first 640 samples of the 1024-point inverse FFT and applies Hann again. It accumulates two waveform sums and a shared squared-window denominator, then releases finalized hops:

$$
\hat{x}_s[n]=\frac{\sum_t w[n-tH+320]\,\mathrm{IFFT}(Y_s[t])[n-tH+320]}
{\max(\epsilon,\sum_t w^2[n-tH+320])}.
$$

The controller maps frame output samples to original capture time. During turn changes the router interpolates the output weights over the configured fade. Same-frame estimates continue for both speakers regardless of the authorized route. Physical speaker playback can re-enter the microphone. Playback defaults to zero for new sessions on this branch, and headphones can be used if monitoring is needed.

## Pseudocode

```text
enroll independently or load profiles → cache fixed priors and power/LPC banks
for each captured mono block:
    convert to 16 kHz with continuous filter state
    for each newly available overlapping frame:
        compute complex STFT, energy, ZCR, periodicity and harmonic peaks
        propose joint pitch states and update fixed-lag scores
        fit two calibrated band-power spectra
        when the older frame is resolved:
            smooth its power fits; form harmonic- and LPC-weighted mask ratio
            optionally correct the male mask using the cepstral fit
            apply masks to the complex mixture STFT
            inverse FFT, weighted overlap-add, normalize and emit one hop
    route by captured sample time and crossfade at turns
    optionally write aligned WAVs and turn log from the worker
    apply playback gain separately, resample and queue device output
```

## Post-session analysis

[analyze_session.py](../analyze_session.py) runs the same streaming engine outside the live app. Its replay subclass observes frame inputs and the masks returned to synthesis. It creates STACF/pitch, spectral mechanism and recorded spectrogram/STE figures as PNG and PDF. Per-figure equations, measured interpretation, limitations and viva prompts go to the generated summary. The original recorder does not store frame F0 or masks. Saved PCM quantization and clipping can change replay relative to the original live floating-point process. The figure labels identify replay, profile statistics and original WAV or turn-log measurements. A turn log records route changes at sample indices, without a field for distinguishing a timer event from a manual Next action.

```powershell
.\.venv\Scripts\python.exe analyze_session.py "outputs/realtime/sessions/SESSION" --profiles "outputs/realtime/SESSION.profile.json" --male-calibration "male_voice_sample.wav" --female-calibration "female_voice_sample.waw.wav"
```

Use the same profile the session used, and supply the actual isolated enrollment WAVs if they were saved. For live enrollment without archived source WAVs, omit the calibration options to show the saved pitch ranges and mark the missing STACF panels. If using other isolated speech as examples, identify them as examples when presenting the figures. See the [README](../README.md) for recording instructions. All generated analysis files stay under the session's analysis directory.

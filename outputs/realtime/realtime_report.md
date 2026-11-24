# Real-time selective muting development report

## A. Architecture

A single full-duplex PortAudio stream reads one microphone channel and writes one playback channel. A lightweight callback copies float32 samples into a bounded input ring, consumes the output ring, measures peak levels and maintains discontinuity counters. Each single-producer/single-consumer ring publishes its cursor after copying the samples and their capture-time metadata. Each holds up to 250 ms at the hardware rate. Startup and recovery prime at least 20 ms of output. A worker converts hardware audio to 16 kHz, continuously estimates both voices, selects the authorized output against its capture timestamp, applies fixed user gain and clipping protection, resamples, and queues playback. One duplex stream supplies both device clocks. A pair that cannot open together requires another device selection.

Tk polls status every 75 ms. Device operations, file import, calibration and shutdown run outside the GUI event loop. Optional recording uses a separate writer with a bounded 128-packet queue. It writes aligned mixture, selected, female and male WAVs, turn events and frame diagnostics incrementally. A recording backlog or I/O failure stops the session and reports the problem. Recording defaults off. Profiles remain fixed until the operator stops and enrolls again.

## B. Offline-to-streaming changes

| Offline operation | Streaming implementation |
|---|---|
| Centered STFT over a padded recording | Persistent float32 input with left startup padding, 640-sample periodic Hann, 160-sample hop, 1024-point FFT. Frames run only when all their samples have arrived. Only finite replay uses explicit EOF padding. |
| Whole-recording peak-energy gate | Fixed threshold from the quieter enrollment's 95th-percentile frame energy plus the configured quiet offset (−42 dB), with an absolute floor. Energy gates pitch proposals. Speaker-specific profiles and spectral evidence assign energy to the two estimates. |
| Full-sequence joint pitch traceback | Reuse autocorrelation, real cepstrum, refined harmonic peak analysis, calibrated pitch priors and joint candidate emissions. Carry normalized state scores and two hops of backpointers. Freeze an older joint state after this bounded lag. |
| Batched spectral-power fitting and LPC adjustment | Reuse the nonnegative two-template fit on each frame. Cache band geometry, spectral comb and normalized LPC exemplars. Preserve calibrated band-density interpolation, uncertainty floor, harmonic redistribution and LPC adjustment. |
| Symmetric whole-array temporal filters | Smooth fitted powers over up to two past and two future hops already present in the pitch lag. Smooth the final female mask along frequency, then use a causal temporal average (72% current mask). The male path uses the complementary base mask plus the existing cepstral assist when both enrollment banks support it. |
| Batch inverse STFT | Persistent two-channel weighted overlap-add and squared-Hann normalization. Emit each finalized hop once at its original sample index. Bypass reconstructs both channels to float32 precision, including the first and final partial hops. |
| Full-file rate conversion and offline gain choices | Causal rational polyphase FIRs retain filter history and rational phase across callbacks. Live output uses a volume setting and a hard peak bound. Controlled-mixture normalization and listening-copy loudness matching stay in the offline evaluator. |
| Whole-file diagnostics and reference labels | Only the evaluator reads isolated test references. Runtime history comprises bounded frame, audio and telemetry buffers. The optional writer streams to disk. |

New enrollments include the 2 ms low-quefrency spectral bank recommended for male listening comparisons. Compatible historical band-power profiles use the current baseline mask if that bank is absent. The independent 64 ms male-window candidate remains an offline experiment.

## C. Audio quality

Replay used the original aligned PCM references and the same profiles for streaming and offline comparisons. Samples entered the engine in a repeating sequence of 1, 127, 320, 79, 512 and 160 samples. The engine withholds output until ready, returns samples in original capture order and drains the final partial hop. Scores therefore compare matching capture indices without correlating against references to guess a delay. A separate test checks that the first 160-sample hop becomes available after exactly 960 captured samples.

Configuration selection used only development and validation at 0 dB. Male/female SI-SDR improvements on validation were 8.13/8.22 dB with past-only band smoothing and 20 ms pitch lag, and 8.20/8.31 dB with two future band frames inside the same lag. Development improved from 8.22/8.19 to 8.32/8.32 dB. Extending bounded tracking to 30 or 50 ms changed validation scores by under 0.01 dB per speaker. The final run used the 20 ms setting across all five strict held-out ratios.

| Strict test, mean across ratios | Male SI-SDRi | Female SI-SDRi | Male target gain | Female leakage attenuation in male estimate | Male 2.5–4 kHz target gain | Male envelope shape RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Offline current | 5.82 | 5.80 | −1.15 | 12.60 | −11.10 | 10.65 |
| Offline cepstral-assisted | 6.14 | 5.80 | −1.11 | 12.36 | −9.90 | 10.29 |
| Streaming cepstral-assisted | 6.08 | 5.69 | −1.11 | 12.35 | −9.76 | 10.28 |

All values are in dB. Target gain and interferer attenuation use joint regression on both references. Band gain applies the same band filter to estimated and reference waveforms before regression. Envelope shape RMSE uses the existing 2 ms cepstral diagnostic on active male-reference frames. Female streaming target gain was −2.33 dB and male leakage attenuation in that channel was 18.23 dB. The measurements support close preservation of the tested male envelope and consonant-band energy. Intelligibility and preference still need listening scores.

| Input female/male level (dB) | Streaming male SI-SDRi (dB) | Streaming female SI-SDRi (dB) |
|---:|---:|---:|
| −10 | 2.58 | 9.72 |
| −5 | 5.62 | 8.95 |
| 0 | 7.21 | 6.98 |
| +5 | 7.78 | 3.84 |
| +10 | 7.22 | −1.05 |

In development single-speaker WAVs, the unauthorized output was 11.41 dB below the female-only input and 14.57 dB below the male-only input (RMS). The authorized channels lost 1.64 and 0.83 dB respectively. The local quality CSVs include scores by ratio and five bands from 80 Hz to 8 kHz.

## D. Latency and throughput

**Replay.** Across the five strict conditions, real-time factor was 0.349–0.399. Per-run median frame time was 3.24–3.76 ms, p95 was 4.86–5.75 ms, p99 was 5.96–6.65 ms and the largest observed frame was 14.05 ms. Four of 9,580 frames exceeded the 10 ms deadline. Offline replay ran as fast as possible, using about 95–97% of one CPU core. Process RSS (including evaluator, references and offline analyses) was 231–237 MiB.

**Device run.** Windows WASAPI opened the AMD microphone array and Realtek output as one 48 kHz stream, with float32 audio converted to/from 16 kHz. Enumeration exposed ten Windows endpoints across host APIs. Of the probed rates only 48 kHz passed both settings checks on this pair. Playback gain was zero, the input was quiet and recording was disabled. A 4 s start/stop check preceded a fresh 300.02 s run.

| Five-minute device run | Observed |
|---|---:|
| DSP frames / 10 ms deadline misses | 30,000 / 36 |
| Frame median / p95 / p99 / maximum, trailing 10,000 frames | 4.21 / 6.34 / 7.70 / 28.04 ms |
| Output underflows / overflows | 2 / 0 |
| Input / output high-water occupancy | 2,400 / 2,880 hardware samples (50 / 60 ms) |
| Each ring capacity | 12,000 hardware samples |
| Process CPU, one-core equivalent | 51.7% |
| RSS first / last / maximum | 111.38 / 112.19 / 112.20 MiB |
| Estimated ADC-to-DAC latency, sampled median / p95 / maximum | 101.4 / 125.0 / 130.1 ms |

**Latency budget.** For the earliest sample of a finalized hop, the capture horizon is 40 ms of analysis plus 20 ms of pitch lookahead. Smoothing fits within that lag. At 48 kHz each converter contributes 0.625 ms group delay. A conservative planning sum using driver-reported input/output delays (22 + 32 ms), 60 ms DSP horizon, 1.25 ms FIR delay, 20 ms playback priming, 4.21 ms median DSP time, up to a 10 ms example device period and 1 ms worker poll is approximately **150.5 ms**. These terms may overlap and queue occupancy varies. The 20 ms crossfade changes the balance between estimates during a switch while playback continues.

The device estimate follows the capture timestamp carried with the output samples to the driver's scheduled DAC time. Earlier startup testing with a single clock anchor produced a spurious ~1.4 s estimate. The interface now carries timestamps through the rings, and shows a budget if the host omits its timestamps. The 101.4 ms figure is a driver-clock estimate. A synchronized electrical or acoustic loopback measurement remains outstanding.

## E. Debate controller

A monotonic clock expressed in 16 kHz sample ticks drives IDLE, both calibration states, READY, the two turns, PAUSED and STOPPED. Automatic boundaries derive from the original deadline and exact integer durations. Recent timestamped control changes allow queued samples to use the decision in effect at capture. The router splits a block at boundaries and carries crossfade weights across calls. Both speaker estimates run continuously through a switch.

Manual selection freezes automatic alternation. Both uses the original mixture. Mute and pause silence playback. Resume retains remaining time and discards queued pre-resume sound. Finite-round completion lets already-routed output reach the device. A manual stop gates playback immediately.

In the deterministic two-round replay, the authorized speaker changed at samples **48,000, 112,000 and 160,000** and stopped at **224,000** (3, 7, 10 and 14 seconds). Outside the 320-sample transitions the selected waveform matched the requested estimate. Interferer gains across the four turns were −14.83, −17.60, −15.46 and −16.67 dB. The saved WAVs, turn CSV and transition plot are under `outputs/realtime/debate/`.

## F. Test results

- **36 automated tests passed, zero failed.** All 22 original tests passed on the tagged baseline and development branch. Added coverage includes exact capture horizon, fixed-lag history, future-prefix independence, bypass reconstruction, block invariance, silence, unequal levels, single voices, bounded rings, converter phase, recording drain, enrollment validation, sample-time routing, crossfades, pause/resume, manual and automatic turns, restart and stream closure.
- All five offline CLI commands (`inspect`, `calibrate`, `evaluate`, `separate`, `visualize`) completed end to end using temporary synthetic WAVs. Both original isolated WAVs also passed the live-import enrollment checks.
- The physical microphone enrollment recorder captured 30.0 s, reached zero on its countdown, and rejected the quiet room for insufficient active speech. No audio was retained. A Tk integration check opened the Windows devices, exercised pause/resume and manual/automatic selection, completed a limited round, and closed the stream.
- Physical acceptance still requires two enrolled people speaking on cue, audible headphone playback, full debate-duration testing, and a synchronized loopback reference. Actual device unplugging was not exercised. Tests inject a capture discontinuity to check error handling.

## G. Remaining limitations

Classical single-channel masks retain mixture phase. Calibration mismatch, overlapping harmonics, consonants and unrelated sounds can leak into either estimate. Even the louder female speaker lost SI-SDR relative to the input at +10 dB. The male output still attenuated 2.5–4 kHz substantially. The five-minute quiet hardware run does not establish intelligibility or dropout-free operation under sustained overlapping speech. Python scheduling can exceed the hop deadline. The recorder writes standard PCM WAVs and is subject to filesystem space and RIFF size limits.

Use headphones or isolated playback. Acoustic echo cancellation is outside this prototype. Disable hardware and OS microphone monitoring to avoid a second audio path. For physical latency measurement, use an isolated split reference and record the reference and analog output on synchronized channels while replaying broadband pulses in Both mode. Measure their sample offset at multiple times and divide by the capture rate. Use safe line levels and keep the reference outside the public output loop.

## H. Launch and reproduce

From the project root in a Python environment with Tkinter:

```powershell
python -m pip install -r requirements.txt
python realtime_app.py
python -m unittest discover -s tests -v
python streaming_evaluation.py select
python streaming_evaluation.py final
python streaming_evaluation.py debate
python hardware_check.py --devices
```

Choose microphone and headphones, enroll or import each isolated voice (or load existing profiles), configure turns and press Start. Recording is opt-in. Use the [README](../../README.md) for fixture generation, physical acceptance steps and troubleshooting. The development changes are confined to `realtime-development`. The stable offline tag identifies the original CLI.

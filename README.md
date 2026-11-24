# Pre-Calibrated Single-Channel Speech Separation Using Classical DSP

This EC4143 project uses one microphone to record two enrolled speakers. The app estimates each voice and routes the selected speaker during the debate.

## Review the four sessions

Open the [session guide](outputs/realtime/sessions/README.md), choose a session, and listen to the original recording before playing the processed output. Each guide also links to both voice estimates and the three analysis PDFs.

The WAVs, figures, turn timings and voice calibration profiles are available in the local project. Git tracks the review guides and source code.

### Start on Windows

Use Python with Tkinter, one microphone, and headphones on a compatible full-duplex device pair. Disable Windows **Listen to this device**, direct microphone monitoring, and any loopback input. Keep speakers acoustically isolated if headphones are unavailable. Lower gain reduces level but cannot remove acoustic feedback. The application performs no echo cancellation.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe realtime_app.py
```

With an activated environment the launch command is `python realtime_app.py`. Use **Refresh** and select the input and output. The device number identifies each endpoint in `python hardware_check.py --devices`. Prefer devices from one host API, for example a WASAPI microphone and WASAPI headphones. **Test input + quiet tone** plays a short tone and reports microphone peak. Debate playback starts muted. Keep playback volume at zero when recording with the laptop microphone and speakers. For live monitoring, use headphones and raise the volume gradually.

1. **Enroll both speakers.** Select 30, 45, 60, or 90 seconds and use **Record / recalibrate**. Speak alone in the same position and room as the debate. Watch the input meter and countdown. An enrollment with too little active or reliably pitched speech, a short capture, or excessive clipping prompts a retry. **Import WAV** accepts an isolated recording, averages channels and resamples to 16 kHz. Import uses the entire enrollment file. Keep evaluation references separate.
2. **Reuse profiles.** Save both locally as a `*.profile.json` file, or load a compatible 40 ms offline profile with calibrated band-power exemplars. New enrollments include the 2 ms cepstral bank for the male estimate. The application uses both fixed banks for the entire session. To replace a speaker, stop and recalibrate that slot.
3. **Configure turns.** Enter positive durations, the starting speaker, a crossfade from 0 to 100 ms (default 20), and rounds (0 means continuous). **Start** opens the full-duplex stream. Automatic mode alternates at the scheduled times, including during silence. **Next speaker** advances one turn and restarts its countdown. One round comprises two turns.
4. **Moderate manually.** Male, Female, Both and Mute switch to manual routing. Both passes the captured mixture. Mute silences the public output. The timer freezes in manual mode. Enabling automatic mode continues the displayed speaker with its remaining time. Pause silences playback and freezes time while the separator continues tracking. Resume discards pre-resume queued playback and continues the turn. Stop closes the device. Start opens a new session.
5. **Record only when wanted.** Enable **Record next session locally** before starting. Four aligned 16 kHz PCM WAVs, turn sample indices and per-block diagnostics go under `outputs/realtime/sessions/`. The selected track is saved at its normal level regardless of the playback volume. The debate Mute and Pause controls still silence the selected track. Device underruns, hardware gain, and emergency playback gates are downstream of this track. Enrollment audio stays in memory unless you separately save its derived profile. Keep these files local.

The DSP requires at most 60 ms of input beyond the start of an emitted hop. Device periods, conversion, computation, and output buffering add latency. The display uses driver timestamps when available. A physical loopback measurement requires a separate reference capture. See the [development report](outputs/realtime/realtime_report.md) for quality, timing, and device tests.

### Analyze a saved debate

Run the analyzer to regenerate the graphs using the session folder and the profile used during the debate. Pass the isolated enrollment WAVs when available to include their short-time autocorrelation panels.

```powershell
python analyze_session.py "outputs/realtime/sessions/SESSION" --profiles "outputs/realtime/SESSION.profile.json" --male-calibration "male_voice_sample.wav" --female-calibration "female_voice_sample.waw.wav"
```

The session's `analysis/` folder receives three figures as PNG and PDF, a metrics CSV, and a summary with viva questions. See the [DSP architecture and equations](docs/realtime_dsp_architecture.md).

## Technical reference

- [Word report handoff and demo plan](docs/EC4143_DSP_Report_Handoff.docx)
- [DSP architecture, equations and analyzer](docs/realtime_dsp_architecture.md)
- [Real-time development and validation report](outputs/realtime/realtime_report.md)
- [Offline optimization report](docs/optimization_report.md) and [intelligibility study](docs/male_intelligibility_report.md)

The offline prototype is available at tag `v0.1.0-offline`. The detailed legacy README and archived experiment files are stored locally under `outputs/.archive/`.

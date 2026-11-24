# Single Microphone Speech Separation

This EC4143 project uses one microphone to capture a two-person debate. The app estimates both voices and lets a timer or moderator choose which one to play. It uses local voice samples and runs on the laptop without a trained neural model.

## Run the app on Windows

Use Python with Tkinter, a microphone, and headphones if you want to hear the live output.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe realtime_app.py
```

1. Select your microphone and output device.
2. Record or import a short solo recording for each speaker. You can also load an existing local profile.
3. Set the turn length. Click **Start** and use **Next speaker** to advance manually.
4. To save the debate, enable **Record next session locally** before starting. Playback starts muted. Use headphones if you raise the volume.

The app saves the original mixture, both voice estimates, and the selected debate output under `outputs/realtime/sessions/`. You can record with playback volume at zero.

## Review the project

- [Micro-project report](docs/Group%23Report.pdf) gives the method, results, and references.
- [Real-time code PDF](docs/submission/Group%23Code.pdf) includes the runnable Python source as an attachment.
- [Combined analysis PDF](docs/submission/Group%23Results.pdf) contains the plots for all four sessions.
- [Teacher submission ZIP](docs/submission/Group%23Submission.zip) contains the three PDFs and a separate folder for the WAV recordings.
- [Four session guides](outputs/realtime/sessions/README.md) explain which tracks and figures to compare. The recordings and figures are available on GitHub.
- [Technical notes](docs/realtime_dsp_architecture.md) cover the formulas and how to regenerate the plots. [Development results](outputs/realtime/realtime_report.md) document the controlled tests and device measurements.
- [Older offline results](docs/offline-results/) accompany the [optimization report](docs/optimization_report.md). The original offline release is tagged `v0.1.0-offline`.

Run `python -m unittest discover -s tests` with the project environment active. Voice profiles and run logs are stored locally.

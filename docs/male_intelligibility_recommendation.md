# Which male output to audition

**Start with the current output A, then compare F and W64 on the same phrase.** F is the strongest candidate for a first listening test because its envelope and separation gains repeated across development, validation and strict test. W64 is the strongest objective result on the strict test. D offers a conservative alternative if leaked female speech is the main distraction. Keep the current global default until listening establishes a preference on the physical recording.

| Choice | What the strict five-ratio test showed | Physical recording, loudness matched |
|---|---|---|
| A · current | Reference for the reported muffling. Mean male SI-SDRi 5.82 dB. | [Play A](listening/real/male_A_current.wav) |
| F · cepstral assist | Mean male SI-SDRi 6.14 dB. Spectral-envelope shape error fell 0.36 dB, with 0.24 dB less overall female attenuation. | [Play F](listening/real/male_F_cepstral_assisted.wav) |
| W64 · longer male window | Mean male SI-SDRi 6.74 dB. Retained 2.17 dB more male power at 2.5–4 kHz and increased overall female attenuation by 0.41 dB. Slower, and weaker than A on development. | [Play W64](listening/real/male_W_64ms.wav) |
| D · reduced smoothing | Smallest practical change among the shortlisted candidates. Recovered 0.34 dB at 2.5–4 kHz while increasing overall female attenuation by 0.39 dB. | [Play D](listening/real/male_D_reduced_smoothing.wav) |

For a stronger presence-band comparison, H recovered 2.18 dB at 2.5–4 kHz but reduced overall female attenuation by 1.41 dB. E’s bounded floor recovered 1.74 dB in that band at a 2.52 dB cost in female attenuation. A brighter sound can come from either voice. [Play H](listening/real/male_H_presence_gain.wav) · [Play E](listening/real/male_E_conservative_mask.wav)

## Listening procedure

1. Use the loudness-matched WAVs directly inside each set. The `raw/` subdirectory contains the unadjusted outputs. Keep playback volume fixed. Use the original mixture to locate a phrase with simultaneous speech, then replay the same time interval in A, F and W64. [Physical mixture](listening/real/original_mixture.wav)
2. Score male intelligibility and naturalness from 1 (poor) to 5 (good). Score female leakage from 1 (barely audible) to 5 (prominent). Note the timestamp and any word ending, vowel or consonant that becomes easier to follow. A second pass in reverse or randomized order can help with order effects. [Physical listening sheet](listening/real/listening_sheet.csv)
3. Repeat on the controlled 0 dB mixture, where the isolated male reference can resolve whether newly audible detail belongs to him. [Isolated male](listening/controlled_0dB/reference_male.wav) · [A](listening/controlled_0dB/male_A_current.wav) · [F](listening/controlled_0dB/male_F_cepstral_assisted.wav) · [W64](listening/controlled_0dB/male_W_64ms.wav) · [Controlled listening sheet](listening/controlled_0dB/listening_sheet.csv)

## Running a candidate

The supplied alternate profiles use the same recordings and calibration ranges as the baseline. The following commands process the physical recording and select the male output. They write independent output folders.

```powershell
python main.py separate --input test.wav --profiles outputs/male_intelligibility/comparisons/profiles/test_640.json --male-variant cepstral_assisted --mode mute-female --out-dir outputs/audition_F
python main.py separate --input test.wav --profiles outputs/profiles.json --male-window-profiles outputs/male_intelligibility/comparisons/profiles/test_1024.json --mode mute-female --out-dir outputs/audition_W64
python main.py separate --input test.wav --profiles outputs/profiles.json --male-variant reduced_smoothing --mode mute-female --out-dir outputs/audition_D
```

**Next development step.** Use the timestamps and ratings to identify which candidate recovers actual male speech cues on the physical mixture. Confirm that preference on a second recording session with synchronized close-microphone references. If W64 wins, measure its 32 ms analysis lookahead and dual-pass cost before considering live processing. If F wins, validate its calibration examples on new phonemes and recording conditions. [Diagnosis and complete results](diagnosis_report.md)

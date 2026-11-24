"""Numerical checks for the offline pipeline and aligned evaluation."""

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import wavfile

from config import DEFAULT, METHODS, ROOT
from src.audio_io import partition, read_audio, write_audio
from src.calibration import band_starts, create_profiles, load_profiles, write_json
from src.evaluation import HEADROOM, controlled_wavs, evaluate, projection_gains, score, si_sdr
from src.pitch_tracking import Tracks, frames, inverse_stft, single_pitch, track
from src.separator import analyze, match_band_powers, select, separate

CFG = DEFAULT


def harmonic(f0: float, seconds: float = 1.3) -> np.ndarray:
    time = np.arange(round(CFG.rate * seconds)) / CFG.rate
    return sum(0.17 / n ** 0.7 * np.sin(2 * np.pi * n * f0 * time) for n in range(1, 11))


def known_profiles() -> dict:
    return {
        "settings": asdict(CFG), "version": 1,
        **{speaker: {
            "pitch": {"median_hz": f0, "log2_scale": 0.1},
            "lpc": {"log_envelopes": [np.zeros(CFG.fft // 2 + 1).tolist()]},
            "recording": {"sha256": "0" * 64, "processing_rate_hz": CFG.rate,
                          "calibration_samples": [0, 2], "held_out_samples": [4, 6]},
        } for speaker, f0 in (("female", 235), ("male", 112.5))},
    }


class SignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = known_profiles()

    def test_weak_pitch_survives_the_dominant_comb(self) -> None:
        x = harmonic(235, .48) * .2 + harmonic(112.5, .48)
        data = frames(x, CFG)
        for gain in (.1, 1):
            result = track(frames(gain*x, CFG), self.profiles, CFG)
            pair = np.abs(result.f0[4:-4] - [235, 112.5]) < [6, 4]
            self.assertGreater(pair.all(1).mean(), .9)
        old = track(data, self.profiles, CFG, legacy=True)
        self.assertLess((old.f0[4:-4] > 0).all(1).mean(), .1)

    def test_power_fit_interior_boundary_silence_and_collinear_examples(self) -> None:
        banks = [np.array([[.8,.2,0], [.1,.1,.8]]), np.array([[0,.2,.8], [.1,.1,.8]])]
        mix = np.array([2*banks[0][0]+3*banks[1][0], banks[1][0], np.zeros(3), banks[0][1]])
        estimated, fit = match_band_powers(mix, banks)
        np.testing.assert_allclose(estimated.sum(1), mix, atol=1e-12)
        np.testing.assert_allclose(estimated[0], [2*banks[0][0], 3*banks[1][0]], atol=1e-12)
        self.assertTrue((estimated >= 0).all())
        np.testing.assert_allclose(fit, [1,1,0,1], atol=1e-12)

    def test_continuity_allows_a_weak_moving_pitch(self) -> None:
        time = np.arange(12800) / CFG.rate
        pitch = 230 + 15 * np.sin(2*np.pi*5*time)
        phase = np.cumsum(pitch) * (2*np.pi/CFG.rate)
        moving = sum(.17 / k**.7 * np.sin(k*phase) for k in range(1, 11))
        data = frames(harmonic(112.5, .8) + .25*moving, CFG)
        centers = np.arange(4, len(data.energy)-4)
        truth = pitch[centers * CFG.hop]
        tracks = track(data, self.profiles, CFG)
        self.assertGreater((np.abs(tracks.f0[centers, 0] - truth) < 7).mean(), .90)

    def test_unpitched_bins_follow_calibrated_energy(self) -> None:
        x = harmonic(235)
        analysis = analyze(x, self.profiles, CFG)
        zeros = np.zeros_like(analysis.tracks.f0)
        analysis = replace(analysis, tracks=Tracks(zeros, zeros), likelihood=analysis.likelihood*0)
        power = np.add.reduceat(np.abs(analysis.frames.spectrum[20])**2, band_starts(CFG))
        example = power/power.sum()
        self.profiles["female"]["band_power_examples"] = [example.tolist()]
        self.profiles["male"]["band_power_examples"] = [example[::-1].tolist()]
        for method in METHODS:
            result = separate(analysis, len(x), self.profiles, CFG, method)
            self.assertGreater(np.linalg.norm(result.audio[0]), 4*np.linalg.norm(result.audio[1]))
            np.testing.assert_allclose(result.audio.sum(0), x, atol=1e-12, rtol=0)

    def test_stft_round_trip_including_edges_and_partial_hop(self) -> None:
        rng = np.random.default_rng(814)
        for length in (1, 159, 160, 641, 4917):
            with self.subTest(length=length):
                x = rng.uniform(-1, 1, length)
                recovered = inverse_stft(frames(x, CFG).spectrum, length, CFG)
                self.assertEqual(len(recovered), length)
                np.testing.assert_allclose(recovered, x, rtol=0, atol=1e-12)

    def test_silence_and_mute_controls(self) -> None:
        x = np.zeros(1001)
        a = analyze(x, self.profiles, CFG)
        self.assertFalse(np.any(a.frames.active))
        self.assertFalse(np.any(a.tracks.f0))
        self.assertFalse(np.any(a.tracks.confidence))
        for method in METHODS:
            y = separate(a, len(x), self.profiles, CFG, method)
            self.assertTrue(np.array_equal(y.audio, np.zeros((2, len(x)))))
            self.assertTrue(np.isfinite(y.masks).all())
        original = np.arange(5)
        estimates = np.stack((original * 0.75, original * 0.25))
        for mode, reference in (("mute-female", estimates[1]), ("mute-male", estimates[0]),
                                ("both", original), ("mute-both", original * 0)):
            np.testing.assert_array_equal(select(original, estimates, mode), reference)

    def test_single_pitch_and_speaker_selection(self) -> None:
        for i, pitch in enumerate((235, 112.5)):
            with self.subTest(pitch=pitch):
                x = harmonic(pitch)
                a = analyze(x, self.profiles, CFG)
                isolated, cep, confidence = single_pitch(a.frames, CFG)
                self.assertLess(abs(np.median(isolated[isolated > 0]) - pitch), 2)
                self.assertTrue(np.any(cep > 0))
                self.assertGreater(np.median(confidence), 0.8)
                self.assertGreater(np.mean(np.abs(a.tracks.f0[:, i] - pitch) < 5), 0.90)
                y = separate(a, len(x), self.profiles, CFG, "harmonic").audio
                self.assertGreater(np.linalg.norm(y[i]), 2 * np.linalg.norm(y[1 - i]))

    def test_dual_pitch_and_reproducibility(self) -> None:
        x = harmonic(235) + harmonic(112.5)
        a = analyze(x, self.profiles, CFG)
        b = analyze(x, self.profiles, CFG)
        np.testing.assert_array_equal(a.tracks.f0, b.tracks.f0)
        np.testing.assert_array_equal(a.tracks.confidence, b.tracks.confidence)
        pitch_ok = np.abs(a.tracks.f0 - [235, 112.5]) < 6
        self.assertGreater(pitch_ok.all(axis=1).mean(), 0.8)
        for method in METHODS:
            first = separate(a, len(x), self.profiles, CFG, method)
            second = separate(b, len(x), self.profiles, CFG, method)
            np.testing.assert_array_equal(first.audio, second.audio)
            np.testing.assert_allclose(first.audio.sum(axis=0), x, rtol=0, atol=1e-12)

    def test_overlapping_harmonics_and_different_amplitudes(self) -> None:
        low = harmonic(110)
        high = harmonic(220)
        for gain in (0.316, 1, 3.162):
            with self.subTest(gain=gain):
                x = (low + gain * high) * 0.4
                a = analyze(x, self.profiles, CFG)
                result = separate(a, len(x), self.profiles, CFG, "harmonic")
                self.assertTrue(np.isfinite(result.audio).all())
                self.assertEqual(result.audio.shape, (2, len(x)))
                self.assertTrue(np.all((result.masks >= 0) & (result.masks <= 1)))
                np.testing.assert_allclose(result.masks.sum(axis=1), 1, rtol=0, atol=1e-15)
                np.testing.assert_allclose(result.audio.sum(axis=0), x, rtol=0, atol=1e-12)

    def test_metrics_and_silent_references(self) -> None:
        t = np.arange(1000) * (2 * np.pi / 1000)
        a, b = np.sin(11 * t), np.cos(31 * t)
        refs = np.stack((a, b))
        self.assertAlmostEqual(si_sdr(a + b, a), 0, places=10)
        self.assertEqual(si_sdr(2 * a, a), 120)
        self.assertIsNone(si_sdr(a, np.zeros_like(a)))
        self.assertIsNone(si_sdr(np.zeros_like(a), a))
        m = score(0.5 * a + 0.1 * b, a + b, refs, 0)
        self.assertAlmostEqual(m["target_gain_db"], 20 * np.log10(0.5))
        self.assertAlmostEqual(m["interferer_attenuation_db"], 20)
        self.assertAlmostEqual(m["si_sdri_db"], 20 * np.log10(5))
        self.assertEqual(projection_gains(a, np.stack((a, a))), [None, None])
        self.assertIsNone(score(0.5 * a, a, np.stack((a, 0 * a)), 0)["si_sdri_db"])


class FileTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_wav_validation_mono_resampling_and_duration(self) -> None:
        fake = self.root / "simulated.waw"
        fake.write_bytes(b"not a WAV file")
        with self.assertRaisesRegex(ValueError, "expected a readable"):
            read_audio(fake)
        path = self.root / "stereo.wav"
        time = np.arange(4410) / 44100
        sine = 0.2 * np.sin(2 * np.pi * 400 * time)
        wavfile.write(path, 44100, np.column_stack((sine, sine)).astype(np.float32))
        x, info = read_audio(path)
        self.assertEqual(info["channels"], 2)
        self.assertEqual(len(x), 1600)
        expected = 0.2 * np.sin(2 * np.pi * 400 * np.arange(1600) / 16000)
        np.testing.assert_allclose(x[30:-30], expected[30:-30], rtol=0, atol=5e-4)
        output = self.root / "out.wav"
        write_audio(output, x, CFG.rate)
        np.testing.assert_allclose(read_audio(output)[0], x, rtol=0, atol=1e-7)

    def test_pcm_mixtures_sum_exactly_and_hit_ratios_after_headroom(self) -> None:
        x = np.stack((harmonic(235), 0.16 * harmonic(112.5)))
        # Add a narrow peak to force common peak attenuation.
        x[0, 0] = 5
        for ratio in (-10, -5, 0, 5, 10):
            with self.subTest(ratio=ratio):
                folder = self.root / str(ratio)
                mix, refs, manifest = controlled_wavs(folder, x, ratio, CFG)
                self.assertEqual(wavfile.read(folder / manifest["mixture"])[1].dtype, np.dtype("int32"))
                self.assertTrue(np.array_equal(mix, refs.sum(axis=0)))
                self.assertLessEqual(max(np.max(np.abs(mix)), np.max(np.abs(refs))), HEADROOM + 1e-9)
                self.assertLess(manifest["common_headroom_gain"], 1)
                self.assertAlmostEqual(manifest["measured_female_to_male_db"], ratio, places=6)
                recovered = x * np.array(manifest["normalization_gains"])[:, None] * manifest["common_headroom_gain"]
                np.testing.assert_allclose(refs, recovered, rtol=0, atol=0.5 / 2 ** 31)

    def test_disjoint_partition_and_profile_provenance(self) -> None:
        p, q = self.root / "female.wav", self.root / "male.wav"
        for path, first, second in ((p, 235, 105), (q, 112.5, 300)):
            write_audio(path, np.r_[harmonic(first, 4.2), harmonic(second, 2.8)], CFG.rate)
        calibration, held, info = partition(p, CFG)
        self.assertEqual(len(calibration), int(4.2 * CFG.rate))
        self.assertEqual(info["held_out_samples"][0] - len(calibration), int(0.25 * CFG.rate))
        self.assertGreater(np.linalg.norm(held), 0)
        path = self.root / "profiles.json"
        create_profiles((p, q), path, CFG)
        fixed = path.read_bytes()
        profiles, cfg = load_profiles(path)
        self.assertEqual(cfg, CFG)
        self.assertAlmostEqual(profiles["female"]["pitch"]["median_hz"], 235, delta=2)
        self.assertAlmostEqual(profiles["male"]["pitch"]["median_hz"], 112.5, delta=2)
        write_audio(p, np.r_[harmonic(235, 4.2), harmonic(110, 2.8)], CFG.rate)
        with self.assertRaisesRegex(ValueError, "does not match the fixed female profile"):
            evaluate((p, q), p, None, path, self.root / "evaluation")
        self.assertEqual(path.read_bytes(), fixed)
        write_audio(p, harmonic(235), CFG.rate)
        with self.assertRaisesRegex(ValueError, "too short"):
            partition(p, CFG)

    def test_resampling_never_crosses_calibration_boundary(self) -> None:
        path = self.root / "boundary.wav"
        source_rate = 44100
        cfg = replace(CFG, guard_seconds=0)
        raw = np.r_[np.zeros(3 * source_rate), np.ones(2 * source_rate)]
        write_audio(path, raw, source_rate)
        calibration, held, meta = partition(path, cfg)
        self.assertFalse(np.any(calibration))
        self.assertEqual(meta["calibration_samples"][1], meta["held_out_samples"][0])
        self.assertEqual(len(held), 2 * CFG.rate)
        self.assertAlmostEqual(float(np.mean(held[80:-80])), 1, places=5)

    def test_missing_and_incomplete_profiles(self) -> None:
        path = self.root / "profiles.json"
        with self.assertRaisesRegex(ValueError, "run calibrate"):
            load_profiles(path)
        profiles = known_profiles()
        del profiles["male"]["lpc"]
        write_json(path, profiles)
        with self.assertRaisesRegex(ValueError, "run calibrate"):
            load_profiles(path)


if __name__ == "__main__":
    unittest.main()

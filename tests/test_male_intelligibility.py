"""Regression and alignment tests for the optional male-only masks."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy import fft

from config import DEFAULT as CFG, OUTPUT, ROOT
from main import parser, run
from male_listening import loudness
from male_study import bin_indices, read_case
from src.audio_io import read_audio, write_audio
from src.calibration import band_starts, load_profiles, write_json
from src.cepstrum import log_envelope, real_cepstrum
from src.pitch_tracking import frames, inverse_stft
from src.separator import MALE_VARIANTS, analyze, separate, select, soft_power_ratio
from test_dsp import known_profiles


class MaleMaskTests(unittest.TestCase):
    def test_quefrency_symmetry_and_full_reconstruction(self):
        k = np.arange(CFG.fft//2+1)
        slow = .8*np.cos(2*np.pi*7*k/CFG.fft)
        fast = .2*np.cos(2*np.pi*65*k/CFG.fft)
        magnitude = np.exp(slow+fast)-1e-12
        cep = real_cepstrum(magnitude,CFG)
        np.testing.assert_allclose(cep[[7,-7]], [.4,.4], atol=1e-12)
        np.testing.assert_allclose(log_envelope(magnitude,CFG,1.),slow,atol=1e-12)
        np.testing.assert_allclose(log_envelope(magnitude,CFG,32.),slow+fast,atol=1e-12)
        np.testing.assert_allclose(log_envelope(0*magnitude,CFG,2.),np.log(1e-12))

    def test_soft_power_ratio_zero_extreme_and_gain_invariance(self):
        f = np.array([0.,0.,4.,1e-300,1e300])
        m = np.array([0.,9.,1.,1e-300,1e300])
        np.testing.assert_allclose(soft_power_ratio(f,m,.5), [.5,1,1/3,.5,.5])
        np.testing.assert_allclose(soft_power_ratio(f,m,.5)+soft_power_ratio(m,f,.5),1)
        for invalid in (0,-1,np.nan):
            with self.assertRaises(ValueError):
                soft_power_ratio(f,m,invalid)

    def test_frequency_bands_and_window_center_alignment(self):
        freq = fft.rfftfreq(CFG.fft,1/CFG.rate)
        self.assertTrue(bin_indices(freq,4000,8000)[-1])
        self.assertFalse(bin_indices(freq,2500,4000)[256])
        self.assertTrue(bin_indices(freq,4000,8000)[256])
        x = np.zeros(1391)
        x[320] = .8
        for window in (512,640,1024):
            cfg = replace(CFG,window=window)
            d = frames(x,cfg)
            self.assertEqual(d.samples[2,window//2],x[320])
            np.testing.assert_allclose(inverse_stft(d.spectrum,len(x),cfg),x,atol=1e-12)

    def test_optional_routes_silence_short_input_and_female_protection(self):
        p = known_profiles()
        example = np.arange(1,len(band_starts(CFG))+1,dtype=float)
        for s,v in [('female',example),('male',example[::-1])]:
            for key in ('band_power_examples','cepstral_2ms_power_examples'):
                p[s][key] = [(v/v.sum()).tolist()]
        for x in (np.zeros(1),np.random.default_rng(7).normal(0,.1,721)):
            a = analyze(x,p,CFG)
            base = separate(a,len(x),p,CFG,"harmonic_lpc")
            for variant in MALE_VARIANTS:
                r = separate(a,len(x),p,CFG,"harmonic_lpc",male_variant=variant)
                self.assertEqual(r.audio.shape,(2,len(x)))
                self.assertTrue(np.isfinite(r.audio).all())
                self.assertTrue(((0 <= r.masks) & (r.masks <= 1)).all())
                np.testing.assert_array_equal(r.audio[0],base.audio[0])
                np.testing.assert_array_equal(select(x,r.audio,"mute-female"),r.audio[1])
                if not x.any():
                    self.assertFalse(r.audio.any())

    def test_existing_balanced_fixture(self):
        fixed = OUTPUT/"optimization"/"test"
        if not (fixed/"profiles.json").exists():
            self.skipTest("Historical strict-test fixture unavailable")
        p,c = load_profiles(fixed/"profiles.json")
        x, refs = read_case("test",0)
        self.assertTrue(np.array_equal(refs.sum(0),x))
        baseline = separate(analyze(x,p,c),len(x),p,c,"harmonic_lpc")
        for channel,s in enumerate(("female","male")):
            saved = read_audio(fixed/"ratio_+0dB"/f"output_{s}.wav")[0]
            np.testing.assert_array_equal(baseline.audio[channel].astype(np.float32),saved)

    def test_cli_combines_declared_windows_and_routes_male(self):
        with TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp)
            t = np.arange(4800)/CFG.rate
            x = .2*np.sin(2*np.pi*112*t) + .12*np.sin(2*np.pi*236*t)
            write_audio(folder/"mix.wav",x,CFG.rate)
            x = read_audio(folder/"mix.wav")[0]
            expected = []
            for i,n in enumerate((640,1024)):
                p = known_profiles()
                p["settings"]["window"] = n
                write_json(folder/f"{n}.json",p)
                c = replace(CFG,window=n)
                expected.append(separate(analyze(x,p,c),len(x),p,c,"harmonic_lpc").audio[i].astype(np.float32))
            run(parser().parse_args(["separate","--input",str(folder/"mix.wav"),"--profiles",str(folder/"640.json"),
                                     "--male-window-profiles",str(folder/"1024.json"),"--mode","mute-female",
                                     "--out-dir",str(folder/"run")]))
            for name,y in zip(("female","male","selected"),(*expected,expected[1])):
                np.testing.assert_array_equal(read_audio(folder/"run"/f"output_{name}.wav")[0],y)

    def test_relative_loudness_is_gain_sensitive(self):
        t = np.arange(CFG.rate)/CFG.rate
        tone = .1*np.sin(2*np.pi*1000*t)
        self.assertAlmostEqual(loudness(tone*2,CFG.rate)-loudness(tone,CFG.rate),20*np.log10(2),places=6)

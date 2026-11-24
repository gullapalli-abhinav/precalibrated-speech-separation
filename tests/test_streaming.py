import unittest

import numpy as np

from config import DEFAULT, SPEAKERS
from src.calibration import calibrate_speaker
from src.streaming import StreamingSeparator
from streaming_evaluation import replay
from test_dsp import harmonic


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = {s: calibrate_speaker(harmonic(f), {"file":s}, DEFAULT,
                            include_cepstral=True)[0] for s,f in zip(SPEAKERS,(235,112.5))}

    def test_normalized_overlap_add_edges_and_partial_chunks(self):
        for length in (1, 159, 160, 161, 9997):
            x = np.random.default_rng(length).normal(size=length).astype('float32')
            eng = StreamingSeparator(None, bypass=True)
            y = replay(eng, x)
            for row in y:
                np.testing.assert_allclose(row, x, atol=1e-6)
            with self.assertRaises(ValueError):
                eng.feed(x)

    def test_silence_and_speakers(self):
        for i,f in enumerate((235,112.5)):
            x = np.r_[np.zeros(2000), harmonic(f,.5), np.zeros(1333)]
            y = replay(StreamingSeparator(self.profiles), x)
            self.assertTrue(np.isfinite(y).all())
            self.assertFalse(y[:,:1280].any())
            self.assertGreater(np.linalg.norm(y[i]), 2*np.linalg.norm(y[1-i]))

    def test_block_invariance_fixed_lag_and_capture_horizon(self):
        x = harmonic(235,.5)*.3+harmonic(112.5,.5)
        engine = StreamingSeparator(self.profiles)
        self.assertEqual(engine.delay, 960)
        self.assertEqual(len(engine.feed(x[:959])), 0)
        y = engine.feed(x[959:960])
        self.assertEqual(len(y), 160)
        self.assertEqual(len(engine.tracker.history), 2)
        altered = np.r_[x[:960], -x[960:]]
        y2 = replay(StreamingSeparator(self.profiles), altered)
        np.testing.assert_array_equal(y.T, y2[:,:160])
        fixed = replay(StreamingSeparator(self.profiles), x, (160,))
        variable = replay(StreamingSeparator(self.profiles), x)
        np.testing.assert_array_equal(variable, fixed)

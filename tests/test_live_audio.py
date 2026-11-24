from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import wave

import numpy as np
from scipy.signal import firwin, upfirdn

from config import DEFAULT
from src.calibration import write_json
from src.debate_controller import Controller, RATE, Router
from src.enrollment import enroll, read_session, session_profiles
from src.live_audio import Converter, LiveAudio, Recorder, Ring, record_sample
from test_dsp import harmonic


class DeviceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = {s:enroll(np.tile(harmonic(f),2)) for s,f in (("male",112.5),("female",235))}

    def test_resampler_phase_and_filter_across_variable_blocks(self):
        x = np.random.default_rng(11).normal(size=8001).astype("float32")
        for source,target in ((44100,RATE),(48000,RATE),(RATE,44100),(RATE,48000)):
            c = Converter(source,target)
            factor = max(c.up,c.down)
            h = firwin(20*factor+1,1/factor,window=("kaiser",5))*c.up
            truth = upfirdn(h,x,up=c.up,down=c.down)
            y = np.concatenate([c.feed(chunk) for chunk in np.array_split(x,77)])
            self.assertEqual(len(y),(len(x)*c.up+c.down-1)//c.down)
            np.testing.assert_allclose(y,truth[:len(y)],atol=1e-6,rtol=1e-6)
            self.assertGreater(c.delay_s,0)

    def test_ring_wrap_overflow_and_timestamp(self):
        ring = Ring(5)
        self.assertTrue(ring.put([1,2,3,4],1))
        dest = np.empty(3)
        ring.take(dest)
        self.assertTrue(ring.put([5,6,7,8],2))
        self.assertFalse(ring.put([9],3))
        ring.take(dest)
        np.testing.assert_equal(dest,[4,5,6])
        self.assertAlmostEqual(ring.last_time,1+3/RATE)
        self.assertEqual(ring.take(dest),2)
        self.assertAlmostEqual(ring.last_time,2+2/RATE)

    def test_enrollment_rejection_and_profile_roundtrip(self):
        for x in (np.zeros(RATE*4),np.ones(RATE*4),harmonic(235,.2),
                  np.r_[harmonic(235,.5),np.zeros(RATE*8)], harmonic(235,3)*.00001):
            with self.assertRaises(ValueError):
                enroll(x)
        with self.assertRaises(ValueError):
            enroll(harmonic(235,3),expected_seconds=30)
        with TemporaryDirectory() as d:
            path = Path(d)/"local.profile.json"
            p = session_profiles(self.profiles)
            write_json(path,p)
            self.assertEqual(self.profiles,read_session(path))
            p["male"] = dict(p["male"], band_power_examples=[])
            write_json(path,p)
            with self.assertRaises(ValueError):
                read_session(path)

    def test_capture_countdown_and_device_overflow(self):
        stream = mock.MagicMock()
        stream.read.side_effect = lambda n:(np.full((n,1),.125,'float32'),False)
        sd = mock.MagicMock()
        sd.InputStream.return_value = stream
        with mock.patch("src.live_audio.backend",return_value=sd), mock.patch("src.live_audio.rates",return_value=[48000]):
            levels = []
            from threading import Event
            x, rails = record_sample(0,1,lambda *args:levels.append(args),Event())
            self.assertEqual(len(x),RATE)
            self.assertEqual(levels[-1],(0.,.125))
            self.assertEqual(rails,0.)
            stream.read.side_effect = lambda n:(None,True)
            with self.assertRaises(ValueError):
                record_sample(0,1,lambda *_:None,Event())
            self.assertTrue(stream.__exit__.called)

    def test_callback_public_gate_fault_and_shutdown_restart(self):
        ctl = Controller()
        flags = SimpleNamespace(input_overflow=False,input_underflow=False,output_underflow=False)
        times = SimpleNamespace(inputBufferAdcTime=1.,currentTime=1.,outputBufferDacTime=1.02)
        sd = mock.MagicMock()
        with mock.patch("src.live_audio.backend",return_value=sd), mock.patch("src.live_audio.rates",return_value=[RATE]):
            for _ in range(2):
                a = LiveAudio(self.profiles,ctl)
                a.start(1,2)
                try:
                    a.output.put(np.ones(400),0.)
                    dst = np.empty((257,1),'float32')
                    a.callback(np.zeros_like(dst),dst,len(dst),times,flags)
                    np.testing.assert_equal(dst,1)
                    a.output.put(np.ones(400),0.)
                    a.public = False
                    a.callback(np.zeros_like(dst),dst,len(dst),times,flags)
                    self.assertFalse(dst.any())
                    flags.input_overflow = True
                    a.callback(np.zeros_like(dst),dst,len(dst),times,flags)
                    self.assertTrue(a.error)
                    flags.input_overflow = False
                finally:
                    a.close()
                self.assertFalse(a.thread.is_alive())
            self.assertEqual(sd.Stream.return_value.close.call_count,2)

    def test_worker_preserves_capture_time_and_drains_final_partial(self):
        from src.streaming import StreamingSeparator
        from streaming_evaluation import replay
        ctl = Controller(lambda:0)
        ctl.command("start",(1,1,"male",True,0))
        ctl.command("select","female",tick=1600)
        a = LiveAudio(self.profiles,ctl,fade_ms=0)
        a.configure(RATE)
        a.volume = 1
        x = (harmonic(235,.24)+harmonic(112.5,.24)).astype("float32")[:-1]
        expected = replay(StreamingSeparator(self.profiles),x)
        captured = []
        a.recorder = mock.Mock()
        a.recorder.submit.side_effect = lambda packet:captured.append(packet[0])
        self.assertTrue(a.input.put(x,0.))
        a.epoch = 0.
        a.done.set()
        a._work()
        self.assertFalse(a.error)
        mix,selected,female,male = [np.concatenate(channel) for channel in zip(*captured)]
        np.testing.assert_array_equal(mix,x)
        np.testing.assert_allclose(female,expected[0],atol=1e-6)
        np.testing.assert_allclose(male,expected[1],atol=1e-6)
        np.testing.assert_allclose(selected,np.r_[male[:1600],female[1600:]],atol=1e-6)
        a.volume = .15
        a.source_pos = 0
        a.router = Router(ctl,fade_ms=0)
        a.mixture = x.copy()
        a.recorder.submit.reset_mock()
        a._emit(np.column_stack((expected[0],expected[1])))
        recorded = a.recorder.submit.call_args.args[0][0][1]
        played = a.output.data[:len(a.output)]
        self.assertGreater(np.max(np.abs(recorded)),0)
        np.testing.assert_allclose(played,recorded*.15,atol=1e-7)

    def test_optional_recorder_drains_aligned_tracks(self):
        with TemporaryDirectory() as d, mock.patch("src.live_audio.OUTPUT",Path(d)):
            r = Recorder()
            x = np.zeros(37)
            r.submit(((x,x,x,x),[(0,"MALE_TURN","male",0)],(0,1,0,0,0)))
            r.close()
            for path in r.folder.glob("*.wav"):
                with wave.open(str(path),"rb") as f:
                    self.assertEqual(f.getnframes(),len(x))
            self.assertIn("MALE_TURN",(r.folder/"turns.csv").read_text())

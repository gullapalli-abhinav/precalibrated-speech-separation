"""PortAudio full duplex, bounded sample rings, and stateful rational FIR conversion."""
from collections import deque
from contextlib import ExitStack
import csv
from datetime import datetime
from math import gcd
from queue import Queue, Full, Empty
from threading import Event, Thread
from time import monotonic
import wave

import numpy as np
from scipy.signal import firwin

from config import OUTPUT
from src.debate_controller import RATE, Router
from src.streaming import StreamingSeparator


def backend():
    import sounddevice
    return sounddevice


def devices():
    return list(enumerate(backend().query_devices()))


def rates(input_id, output_id=None):
    sd = backend()
    default = sd.query_devices(input_id, "input")["default_samplerate"]
    for rate in dict.fromkeys((RATE, 48000, 44100, int(default))):
        try:
            sd.check_input_settings(device=input_id, channels=1, dtype="float32", samplerate=rate)
            if output_id is not None:
                sd.check_output_settings(device=output_id, channels=1, dtype="float32", samplerate=rate)
        except sd.PortAudioError:
            continue
        yield rate


class Converter:
    """Rational causal polyphase FIR. Input history and fractional phase survive every call."""
    def __init__(self, source, target):
        factor = gcd(source, target)
        self.up, self.down = target//factor, source//factor
        self.received = self.produced = 0
        self.delay_s = 0.
        if self.up == self.down:
            return
        m = max(self.up, self.down)
        kernel = firwin(20*m+1, 1/m, window=("kaiser", 5.))*self.up
        self.delay_s = (len(kernel)-1)/(2*source*self.up)
        self.phases = np.pad(kernel,(0,(-len(kernel))%self.up)).reshape(-1,self.up).T.astype('float32')
        self.taps = self.phases.shape[1]
        self.history = np.zeros(self.taps, 'float32')

    def feed(self, x):
        x = np.asarray(x,'float32')
        if self.up == self.down or not len(x):
            return x
        buf = np.r_[self.history, x]
        start = self.received
        self.received += len(x)
        stop = (self.received*self.up + self.down-1)//self.down
        t = np.arange(self.produced, stop, dtype=np.int64)*self.down
        self.produced = stop
        windows = np.lib.stride_tricks.sliding_window_view(buf, self.taps)
        y = np.einsum("ij,ij->i", windows[t//self.up-start+1, ::-1], self.phases[t%self.up])
        self.history = buf[-self.taps:].copy()
        return y


class Ring:
    """Single producer and single consumer. Each owns one cursor and publishes it after the copy."""
    def __init__(self, capacity, rate=RATE):
        self.rate = rate
        self.stamps = np.empty(capacity, float)
        self.last_time = 0.
        self.data = np.empty(capacity, 'float32')
        self.read = self.written = self.peak = 0

    def __len__(self):
        return self.written-self.read

    def put(self, x, timestamp=0.):
        size = len(self.data)
        if len(x) > size-len(self):
            return False
        pos = self.written % size
        n = min(len(x), size-pos)
        self.data[pos:pos+n] = x[:n]
        self.data[:len(x)-n] = x[n:]
        intercept = timestamp-self.written/self.rate
        self.stamps[pos:pos+n] = intercept
        self.stamps[:len(x)-n] = intercept
        self.written += len(x)
        self.peak = max(self.peak, len(self))
        return True

    def take(self, dest):
        size = len(self.data)
        count = min(len(dest), len(self))
        pos = self.read % size
        n = min(count, size-pos)
        dest[:n] = self.data[pos:pos+n]
        dest[n:count] = self.data[:count-n]
        if count:
            self.last_time = self.stamps[pos]+self.read/self.rate
        self.read += count
        return count


class Recorder:
    def __init__(self):
        self.folder = OUTPUT/"realtime"/"sessions"/datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.folder.mkdir(parents=True)
        self.queue = Queue(maxsize=128)
        self.done = Event()
        self.error = ""
        self.thread = Thread(target=self._write, daemon=True)
        self.thread.start()

    def submit(self, packet):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.queue.put_nowait(packet)
        except Full as exc:
            raise RuntimeError("Recording worker fell behind; stopped to preserve alignment") from exc

    def _write(self):
        try:
            with ExitStack() as stack:
                files = []
                for name in ("mixture", "selected", "female", "male"):
                    file = stack.enter_context(wave.open(str(self.folder/f"{name}.wav"),"wb"))
                    file.setparams((1,2,RATE,0,"NONE","not compressed"))
                    files.append(file)
                events = csv.writer(stack.enter_context((self.folder/"turns.csv").open("w",newline="")))
                metrics = csv.writer(stack.enter_context((self.folder/"performance.csv").open("w",newline="")))
                events.writerow(("sample","state","route","turn_zero_based"))
                metrics.writerow(("sample","last_frame_ms","deadline_misses","input_samples","output_samples"))
                while not self.done.is_set() or not self.queue.empty():
                    try:
                        audio, transitions, perf = self.queue.get(timeout=.05)
                    except Empty:
                        continue
                    for file, x in zip(files,audio):
                        file.writeframesraw((np.clip(x,-1,1)*32767).astype("<i2").tobytes())
                    events.writerows(transitions)
                    metrics.writerow(perf)
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.done.set()
        self.thread.join()
        if self.error:
            raise RuntimeError(self.error)


class LiveAudio:
    def __init__(self, profiles, controller, fade_ms=20, record=False):
        self.engine = StreamingSeparator(profiles)
        self.router = Router(controller, fade_ms)
        self.record = record
        self.recorder = self.stream = self.thread = None
        self.done = Event()
        self.volume = .15
        self.public = True
        self.play_after = 0
        self.input_level = self.output_level = 0.
        self.underflows = self.overflows = 0
        self.error = ""
        self.epoch = None
        self.latency_ms = None
        self.playing = False
        self.source_pos = 0
        self.end_cursor = None
        self.hardware_timestamps = False

    def start(self, input_id, output_id):
        sd = backend()
        errors = []
        for rate in rates(input_id, output_id):
            try:
                self.stream = sd.Stream(device=(input_id, output_id), samplerate=rate,
                    channels=(1,1), dtype="float32", blocksize=0, latency="low", callback=self.callback)
            except sd.PortAudioError as exc:
                errors.append(str(exc))
                continue
            break
        if self.stream is None:
            raise ValueError("Cannot open a common full-duplex rate. Select devices on one host API. "+" ".join(errors))
        try:
            self.configure(rate)
            self.device_latency = self.stream.latency
            if self.record:
                self.recorder = Recorder()
            self.thread = Thread(target=self._work, daemon=True)
            self.thread.start()
            self.stream.start()
        except BaseException:
            self.close()
            raise

    def configure(self, rate):
        self.rate = rate
        self.input = Ring(rate//4,rate)
        self.output = Ring(rate//4,rate)
        self.down, self.up = Converter(rate,RATE), Converter(RATE,rate)
        self.mixture = np.empty(0,'float32')
        self.prefill = round(.020*rate)
        self.anchors = deque()

    def callback(self, indata, outdata, frames, times, status):
        outdata.fill(0)
        if self.error or self.done.is_set():
            return
        if status.input_overflow or status.input_underflow:
            self.overflows += int(status.input_overflow)
            self.underflows += int(status.input_underflow)
            self.error = "Input discontinuity. Stop and restart the device."
            return
        if status.output_underflow:
            self.underflows += 1
        clock = monotonic()
        self.hardware_timestamps = all(v > 0 for v in
            (times.inputBufferAdcTime, times.outputBufferDacTime, times.currentTime))
        if self.hardware_timestamps:
            adc = clock + times.inputBufferAdcTime-times.currentTime
        else:
            adc = clock-frames/self.rate
        if self.epoch is None:
            self.epoch = adc
        x = indata[:,0]
        self.input_level = float(np.max(np.abs(x), initial=0))
        if not self.input.put(x, adc):
            self.overflows += 1
            self.error = "Input buffer full. Stop and restart the device."
            return
        if not self.playing and len(self.output) < max(self.prefill, min(frames,len(self.output.data))):
            self.output_level = 0
            return
        self.playing = True
        count = self.output.take(outdata[:,0])
        if count < frames:
            self.underflows += 1
            self.playing = False
        if count:
            capture = self.output.last_time
            dac = clock+times.outputBufferDacTime-times.currentTime
            self.latency_ms = 1000*(dac-capture) if self.hardware_timestamps else None
            cutoff = max(0, min(count, int(np.ceil((self.play_after/RATE-capture)*self.rate))))
            outdata[:cutoff] = 0
        if not self.public:
            outdata.fill(0)
        self.output_level = float(np.max(np.abs(outdata), initial=0))

    def _emit(self, estimates, playback=True):
        if not len(estimates):
            return
        mix, self.mixture = self.mixture[:len(estimates)], self.mixture[len(estimates):]
        while len(self.anchors) > 1 and self.anchors[1][0] <= self.source_pos:
            self.anchors.popleft()
        index, timestamp = self.anchors[0]
        tick = round(timestamp*RATE)+self.source_pos-index
        routed, events = self.router.apply(tick, estimates, mix)
        selected = np.clip(routed*self.volume,-1,1)
        if self.recorder is not None:
            self.recorder.submit(((mix,routed,*estimates.T),
                [(self.source_pos+t-tick,*rest) for t,*rest in events],
                (self.source_pos,1000*self.engine.times[-1],self.engine.misses,len(self.input),len(self.output))))
        self.source_pos += len(estimates)
        if playback:
            for t,state,_,_ in events:
                if state == "STOPPED":
                    self.end_cursor = self.output.written + round(((t-tick)/RATE+self.up.delay_s)*self.rate)
            if not self.output.put(np.clip(self.up.feed(selected),-1,1), tick/RATE-self.up.delay_s):
                self.overflows += 1
                raise RuntimeError("Output buffer full. Stop and restart the device.")

    def _work(self):
        buf = np.empty(round(self.rate*.010),'float32')
        try:
            while not self.done.is_set() or len(self.input):
                if self.error:
                    return
                n = self.input.take(buf)
                if not n:
                    self.done.wait(.001)
                    continue
                self.anchors.append((self.engine.received, self.input.last_time-self.down.delay_s))
                x = self.down.feed(buf[:n])
                self.mixture = np.r_[self.mixture,x]
                self._emit(self.engine.feed(x), not self.done.is_set())
            if self.epoch is not None and self.recorder is not None:
                self._emit(self.engine.finish(),False)
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        try:
            if self.stream is not None:
                try:
                    self.stream.abort()
                finally:
                    self.stream.close()
                    self.stream = None
        finally:
            self.done.set()
            if self.thread:
                self.thread.join()
            if self.recorder:
                self.recorder.close()


def record_sample(device, seconds, progress, cancel):
    sd = backend()
    for rate in rates(device):
        try:
            stream = sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32")
        except sd.PortAudioError:
            continue
        break
    else:
        raise ValueError("Cannot open the selected microphone")
    data = np.empty(round(rate*seconds), 'float32')
    pos = 0
    with stream:
        while pos < len(data):
            if cancel.is_set():
                raise ValueError("Recording cancelled")
            n = min(round(.04*rate),len(data)-pos)
            block, overflow = stream.read(n)
            if overflow:
                raise ValueError("Microphone overflow. Retry enrollment")
            data[pos:pos+n] = block[:,0]
            pos += n
            progress((len(data)-pos)/rate,float(np.max(np.abs(block))))
    from src.audio_io import resample
    return resample(data.astype(float),rate,RATE), float(np.mean(abs(data)>=.995))

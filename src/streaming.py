"""Incremental 16 kHz separator. Output columns follow config.SPEAKERS."""
from collections import deque
from time import perf_counter

import numpy as np
from scipy import fft, ndimage

from config import DEFAULT, SPEAKERS
from src.calibration import band_starts
from src.cepstrum import log_envelope
from src.pitch_tracking import EPS, Frames, hann, harmonic_grid, pitch_candidates
from src.separator import match_band_powers, soft_power_ratio


class FixedLag:
    def __init__(self, lag):
        self.lag = lag
        self.history = deque()
        self.previous = None

    def push(self, states, emission, count):
        parents = None
        if self.previous is None:
            scores = emission.copy()
        else:
            prior, scores = self.previous
            delta = np.abs(np.log2(np.maximum(states, 1))[:, None] -
                           np.log2(np.maximum(prior, 1))[None])
            on, was = states[:, None] > 0, prior[None] > 0
            cost = np.where(on & was, np.minimum(1.1*delta + 5*delta**2, 2), .14*(on ^ was))
            paths = scores - .35*cost.sum(-1)
            parents = paths.argmax(1)
            scores = emission + paths[np.arange(len(states)), parents]
        scores -= scores.max()
        self.previous = states, scores
        self.history.append((states, emission, parents))
        if len(self.history) <= self.lag:
            return None
        index = scores.argmax()
        for _, _, parent in list(self.history)[1:][::-1]:
            index = parent[index]
        states, emission, _ = self.history.popleft()
        without = [count*(count+1) + index % (count+1), index//(count+1)*(count+1)+count]
        pitch = states[index]
        conf = np.where(pitch > 0, np.clip((emission[index]-emission[without]+.03)/.34, 0, 1), 0)
        return pitch, conf


class StreamingSeparator:
    """feed emits finalized samples starting at source index zero; finish drains EOF padding."""
    def __init__(self, profiles, cfg=DEFAULT, lag=2, smoothing="bounded", bypass=False):
        if not 0 <= lag <= 5 or smoothing not in ("causal", "bounded"):
            raise ValueError("lag must be 0..5 hops and smoothing causal or bounded")
        self.cfg = cfg
        self.tracker = FixedLag(lag)
        self.lag = lag
        self.radius = min(2, lag) if smoothing == "bounded" else 0
        self.bypass = bypass
        self.profiles = profiles
        self.pending = deque()
        self.past = deque(maxlen=2)
        self.previous_mask = None
        self.window = hann(cfg).astype('float32')
        self.samples = np.zeros(cfg.window//2, 'float32')
        self.ola = np.zeros((cfg.window, 2), 'float32')
        self.norm = np.zeros(cfg.window, 'float32')
        self.start = -cfg.window//2
        self.received = self.emitted = 0
        self.closed = False
        self.times = deque(maxlen=10000)
        self.misses = self.frame_count = 0
        self.freq = fft.rfftfreq(cfg.fft, 1/cfg.rate)
        if bypass:
            return
        self.starts = band_starts(cfg)
        self.widths = np.diff(np.r_[self.starts, len(self.freq)])
        centers = self.starts + (self.widths-1)/2
        self.pos = np.interp(np.arange(len(self.freq)), centers, np.arange(len(centers)))
        self.banks = [np.asarray(profiles[s]["band_power_examples"]) for s in SPEAKERS]
        self.cep_banks = None
        if all("cepstral_2ms_power_examples" in profiles[s] for s in SPEAKERS):
            self.cep_banks = [np.asarray(profiles[s]["cepstral_2ms_power_examples"]) for s in SPEAKERS]
        self.band = (self.freq >= 200) & (self.freq <= 4500)
        self.lpc = []
        for s in SPEAKERS:
            bank = np.asarray(profiles[s]["lpc"]["log_envelopes"])
            shapes = bank[:, self.band].copy()
            shapes -= shapes.mean(1, keepdims=True)
            shapes /= np.maximum(np.linalg.norm(shapes, axis=1, keepdims=True), EPS)
            self.lpc.append((bank, shapes))
        level = min(profiles[s]["ste_dbfs"]["p95"] for s in SPEAKERS)
        self.threshold = max(1e-10, 10**((level + cfg.quiet_db)/10))

    @property
    def delay(self):
        """Maximum capture extent beyond start of an emitted hop, in samples."""
        return self.cfg.window + self.lag*self.cfg.hop

    def _mask(self, spectrum, pitch, confidence, bands, fit):
        density = bands/self.widths
        env = np.stack([np.interp(self.pos, np.arange(len(row)), row) for row in density])
        env += (.02 + .25*(1-fit))*env.mean(0)
        comb = harmonic_grid(pitch, self.freq, self.cfg.rate/self.cfg.fft)**1.7
        strength = env*(1 + confidence[:, None]*(comb/np.maximum(comb.mean(1, keepdims=True), .05)-1))+EPS
        smooth = ndimage.gaussian_filter1d(abs(spectrum), 6)
        obs = np.log(np.maximum(smooth[self.band], smooth.max()*1e-3 + EPS))
        obs += np.log(np.maximum(2*np.sin(np.pi*self.freq[self.band]/self.cfg.rate), .03))
        obs -= obs.mean()
        obs /= max(np.linalg.norm(obs), EPS)
        for i, (bank, shapes) in enumerate(self.lpc):
            k = min(3, len(bank))
            best = np.argpartition(np.maximum(shapes @ obs, 0), -k)[-k:]
            shape = bank[best].mean(0)
            strength[i] *= np.exp(.22*confidence[i]*np.clip(shape-np.median(shape), -2, 2))
        female = ndimage.gaussian_filter1d(strength[0]/strength.sum(0), .55, mode="nearest")
        if self.previous_mask is not None:
            female = .72*female + .28*self.previous_mask
        self.previous_mask = female
        male = 1-female
        if self.cep_banks is not None:
            power = np.exp(2*log_envelope(spectrum[None], self.cfg, 2.))
            parts, quality = match_band_powers(np.add.reduceat(power, self.starts, axis=1), self.cep_banks)
            ratio = soft_power_ratio(*parts[0], 1.)
            male += .35*quality[0]*(np.interp(self.pos, np.arange(len(ratio)), ratio)-male)
        return np.stack((female, male))

    def _frame(self, samples):
        cfg = self.cfg
        spectrum = fft.rfft(samples*self.window, n=cfg.fft)
        if self.bypass:
            self.pending.append(spectrum)
            if len(self.pending) <= self.lag:
                return
            masked = np.broadcast_to(self.pending.popleft(), (2, len(spectrum)))
        else:
            energy = np.mean(samples**2, keepdims=True)
            frame = Frames(samples[None], spectrum[None], energy,
                           np.array([np.mean(np.diff(np.signbit(samples)))]), energy > self.threshold)
            states, emissions, count = pitch_candidates(frame, self.profiles, cfg)
            track = self.tracker.push(states[0], emissions[0], count)
            power = np.add.reduceat(abs(spectrum)**2, self.starts)[None].astype(float)
            bands, fit = match_band_powers(power, self.banks)
            self.pending.append((spectrum, bands[0], fit[0]))
            if track is None:
                return
            spectrum, current, fit = self.pending.popleft()
            context = [*self.past, current] + [v[1] for v in list(self.pending)[:self.radius]]
            offsets = np.arange(len(context))-len(self.past)
            bands = np.average(context, axis=0, weights=np.exp(-.5*(offsets/1.25)**2))
            self.past.append(current)
            masked = spectrum*self._mask(spectrum, *track, bands, fit)
        self.ola += fft.irfft(masked, n=cfg.fft)[:, :cfg.window].T*self.window[:, None]
        self.norm += self.window**2
        hop = cfg.hop
        offset = min(hop, max(0, -self.start))
        out = self.ola[offset:hop]/np.maximum(self.norm[offset:hop, None], EPS)
        for buf in (self.ola, self.norm):
            buf[:-hop] = buf[hop:]
            buf[-hop:] = 0
        self.start += hop
        return out

    def _advance(self, x):
        self.samples = np.concatenate((self.samples, x))
        output = []
        cfg = self.cfg
        while len(self.samples) >= cfg.window:
            start = perf_counter()
            block = self._frame(self.samples[:cfg.window])
            elapsed = perf_counter()-start
            self.times.append(elapsed)
            self.frame_count += 1
            self.misses += elapsed > cfg.hop/cfg.rate
            if block is not None and len(block):
                output.append(block)
            self.samples = self.samples[cfg.hop:]
        return np.concatenate(output) if output else np.empty((0, 2), 'float32')

    def feed(self, x):
        x = np.asarray(x, dtype='float32')
        if self.closed or x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Expected finite mono samples in an open stream")
        self.received += len(x)
        out = self._advance(x)
        self.emitted += len(out)
        return out

    def finish(self):
        if self.closed:
            raise ValueError("Stream already finished")
        self.closed = True
        out = self._advance(np.zeros(self.delay+self.cfg.hop, 'float32'))
        return out[:self.received-self.emitted]

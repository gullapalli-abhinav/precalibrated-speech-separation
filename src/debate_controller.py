"""Debate decisions on a monotonic integer clock shared with ADC capture timestamps."""
from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from threading import RLock
from time import monotonic

import numpy as np

RATE = 16000
TURNS = {"male": "MALE_TURN", "female": "FEMALE_TURN"}


def now_tick():
    return round(monotonic()*RATE)


@dataclass(frozen=True)
class State:
    tick: int
    state: str = "IDLE"
    speaker: str = "male"
    route: str = "mute"
    remaining: int = 0
    turn: int = 0
    auto: bool = True
    durations: tuple = (45*RATE, 45*RATE)  # male, female
    rounds: int = 0

    def duration(self, speaker):
        return self.durations[speaker == "female"]

    def at(self, tick):
        if self.state not in TURNS.values() or not self.auto:
            return replace(self, tick=tick)
        elapsed = tick-self.tick
        if elapsed < self.remaining:
            return replace(self, tick=tick, remaining=self.remaining-elapsed)
        elapsed -= self.remaining
        other = "female" if self.speaker == "male" else "male"
        cycles, elapsed = divmod(elapsed, sum(self.durations))
        turn = self.turn + 1 + cycles*2
        speaker = other
        if elapsed >= self.duration(other):
            elapsed -= self.duration(other)
            turn += 1
            speaker = self.speaker
        if self.rounds and turn >= 2*self.rounds:
            return replace(self, tick=tick, state="STOPPED", route="mute", remaining=0,
                           turn=2*self.rounds-1)
        return replace(self, tick=tick, state=TURNS[speaker], speaker=speaker, route=speaker,
                       turn=turn, remaining=self.duration(speaker)-elapsed)


class Controller:
    def __init__(self, clock=now_tick):
        self.clock = clock
        self.lock = RLock()
        self.anchors = deque([State(clock())], maxlen=256)

    def _lookup(self, tick):
        return next((a for a in reversed(self.anchors) if a.tick <= tick), State(tick))

    def state(self, tick=None):
        tick = self.clock() if tick is None else tick
        with self.lock:
            return self._lookup(tick).at(tick)

    def command(self, action, value=None, *, tick=None):
        with self.lock:
            tick = self.clock() if tick is None else tick
            if tick < self.anchors[-1].tick:
                raise ValueError("Control timestamps must be monotonic")
            s = self.state(tick)
            active = s.state in TURNS.values()
            if action == "start":
                male, female, first, auto, rounds = value
                if (not all(isfinite(v) and round(v*RATE) >= 1 for v in (male, female))
                        or first not in TURNS or not isinstance(rounds, int) or rounds < 0):
                    raise ValueError("Enter positive turn durations and a nonnegative round count")
                durations = round(male*RATE), round(female*RATE)
                s = State(tick, TURNS[first], first, first, durations[first == "female"],
                          0, auto, durations, rounds)
            elif action in ("IDLE", "CALIBRATING_MALE", "CALIBRATING_FEMALE", "READY", "STOPPED"):
                s = replace(s, state=action, route="mute", remaining=0)
            elif action == "pause" and active:
                s = replace(s, state="PAUSED")
            elif action == "resume" and s.state == "PAUSED":
                s = replace(s, state=TURNS[s.speaker])
            elif action == "auto" and (active or s.state == "PAUSED"):
                s = replace(s, auto=bool(value), route=s.speaker if value else s.route)
            elif action == "next" and active:
                other = "female" if s.speaker == "male" else "male"
                if s.rounds and s.turn+1 >= 2*s.rounds:
                    s = replace(s, state="STOPPED", route="mute", remaining=0)
                else:
                    s = replace(s, speaker=other, state=TURNS[other], route=other,
                                turn=s.turn+1, remaining=s.duration(other))
            elif action == "select" and active and value in ("male", "female", "both", "mute"):
                speaker = value if value in TURNS else s.speaker
                s = replace(s, state=TURNS[speaker], speaker=speaker, route=value,
                            remaining=s.duration(speaker), auto=False)
            else:
                return s
            self.anchors.append(s)
            return s

    def spans(self, tick, length):
        # Return decisions for the captured interval, including exact intra-block transitions.
        with self.lock:
            end = tick+length
            spans = []
            while tick < end:
                s = self._lookup(tick).at(tick)
                stop = next((a.tick for a in self.anchors if a.tick > tick), end)
                if s.auto and s.state in TURNS.values():
                    stop = min(stop, tick+s.remaining)
                stop = min(end, stop)
                spans.append((s, stop-tick))
                tick = stop
            return spans


class Router:
    def __init__(self, controller, fade_ms=20):
        if not isfinite(fade_ms) or not 0 <= fade_ms <= 100:
            raise ValueError("Crossfade must be between 0 and 100 ms")
        self.controller = controller
        self.fade = round(fade_ms*RATE/1000)
        self.weights = np.zeros(3)
        self.previous = None
        self.left = 0

    def apply(self, tick, estimates, mixture):
        y = np.zeros(len(mixture), 'float32')
        events = []
        start = 0
        for s, length in self.controller.spans(tick, len(y)):
            route = s.route if s.state in TURNS.values() else "mute"
            marker = (s.state, route, s.turn)
            if marker != self.previous:
                events.append((s.tick, *marker))
                self.previous = marker
                self.left = self.fade
            end = start+length
            target = np.array([route == "female", route == "male", route == "both"], float)
            if route == "mute":
                self.weights = target
                self.left = 0
            else:
                signals = np.column_stack((estimates[start:end], mixture[start:end]))
                n = min(length, self.left)
                if n:
                    ramp = np.arange(1,n+1)[:, None]/self.left
                    weights = self.weights + ramp*(target-self.weights)
                    y[start:start+n] = (signals[:n]*weights).sum(1)
                    self.weights = weights[-1]
                    self.left -= n
                y[start+n:end] = signals[n:] @ target
                if not self.left:
                    self.weights = target
            start = end
        return y, events

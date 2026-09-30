import math
from dataclasses import dataclass

from .math_utils import clamp, lerp
from .pose import Pose


@dataclass(frozen=True)
class Keyframe:
    t: float
    joints: tuple
    root_velocity: tuple = (0.0, 0.0)
    omega: float = 0.0


@dataclass(frozen=True)
class Marker:
    """Named point on a clip timeline.

    Pure markers only expose a phase for synchronisation (for example the
    grounded-foot instant shared by walk and run). Markers may also carry an
    event payload; that event fires when playback crosses the marker while
    moving forward.
    """

    name: str
    t: float
    event: object = None

    @property
    def fires_event(self) -> bool:
        return self.event is not None


class AnimationClip:
    """Immutable action resource shared by any number of character instances."""

    def __init__(self, name: str, keyframes, loop: bool, markers=()):
        frames = sorted(keyframes, key=lambda f: f.t)
        if len(frames) < 2:
            raise ValueError(f"clip {name!r} needs at least two keyframes")
        if frames[0].t != 0.0:
            raise ValueError(f"clip {name!r} must start at t=0")
        marks = tuple(sorted(markers, key=lambda m: m.t))
        for marker in marks:
            if marker.t < 0.0 or marker.t > frames[-1].t:
                raise ValueError(
                    f"marker {marker.name!r} of clip {name!r} is outside [0,duration]"
                )
        self.name = name
        self.keyframes = frames
        self.duration = frames[-1].t
        self.loop = loop
        self.joint_count = len(frames[0].joints)
        self.markers = marks

    def _normalise_time(self, t: float) -> float:
        if self.loop:
            return t % self.duration
        return clamp(t, 0.0, self.duration)

    def marker(self, name: str) -> Marker:
        for marker in self.markers:
            if marker.name == name:
                return marker
        raise KeyError(f"clip {self.name!r} has no marker {name!r}")

    def latest_marker_time(self, name: str, cursor: float) -> float:
        """Unwrapped time of the most recent occurrence of *name* at/before cursor.

        For looping clips the marker repeats every period, which is what lets
        walk->run align the grounded-foot phase without a visible foot slide.
        The returned time always satisfies result <= cursor.
        """
        phase = self.marker(name).t
        if self.loop:
            duration = self.duration
            k = math.floor((cursor - phase) / duration + 1e-12)
            return phase + k * duration
        if phase <= cursor + 1e-12:
            return phase
        raise LookupError(
            f"marker {name!r} of non-looping clip {self.name!r} "
            f"has not been reached at cursor {cursor}"
        )

    def events_between(self, start: float, end: float):
        """Event markers strictly crossed while moving start -> end (end > start).

        Returns ((unwrapped_marker_time, Marker), ...) in traversal order.
        Each occurrence of a marker on a looping clip is a distinct event.
        Events are taken on the half-open interval (start, end]: the event
        sitting exactly on the entry cursor was not crossed by this forward
        move and therefore does not fire.
        """
        if end <= start:
            return ()
        fired = []
        for marker in self.markers:
            if not marker.fires_event:
                continue
            phase = marker.t
            if self.loop:
                duration = self.duration
                first_k = math.floor((start - phase) / duration + 1e-12) + 1
                last_k = math.floor((end - phase) / duration + 1e-12)
                occurrences = [
                    (phase + k * duration, marker)
                    for k in range(first_k, last_k + 1)
                ]
            else:
                occurrences = [(phase, marker)] if start < phase <= end else []
            fired.extend(occurrences)
        fired.sort(key=lambda item: item[0])
        return tuple(fired)

    def _segment(self, t: float):
        frames = self.keyframes
        if t >= frames[-1].t:
            return frames[-1], frames[-1], 0.0
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            if a.t <= t <= b.t:
                alpha = (t - a.t) / (b.t - a.t)
                return a, b, alpha
        return frames[0], frames[1], 0.0

    def sample(self, t: float) -> Pose:
        t = self._normalise_time(t)
        a, b, alpha = self._segment(t)
        joints = tuple(lerp(x, y, alpha) for x, y in zip(a.joints, b.joints))
        velocity = tuple(
            lerp(x, y, alpha) for x, y in zip(a.root_velocity, b.root_velocity)
        )
        return Pose(joints, velocity, lerp(a.omega, b.omega, alpha))

    def integrate_root(self, t0: float, dt: float):
        """Exact displacement/turning for piecewise-linear root channels.

        Handles loop wraps by splitting at the wrap point and keyframe times.
        Returns (dx, dy, d_heading) in the clip local frame.
        """
        if dt <= 0.0:
            return 0.0, 0.0, 0.0
        end = t0 + dt
        breaks = {t0, end}
        if self.loop:
            first_wrap = ((t0 // self.duration) + 1) * self.duration
            boundary = first_wrap
            while boundary < end:
                breaks.add(boundary)
                boundary += self.duration
        for frame in self.keyframes:
            marker = frame.t
            if self.loop:
                start_k = int(t0 // self.duration) - 1
                for k in range(start_k, int(end // self.duration) + 2):
                    candidate = k * self.duration + marker
                    if t0 < candidate < end:
                        breaks.add(candidate)
            elif t0 < marker < end:
                breaks.add(marker)

        dx = dy = dh = 0.0
        for left, right in zip(sorted(breaks), sorted(breaks)[1:]):
            p0 = self.sample(left)
            p1 = self.sample(right)
            span = right - left
            dx += 0.5 * (p0.root_velocity[0] + p1.root_velocity[0]) * span
            dy += 0.5 * (p0.root_velocity[1] + p1.root_velocity[1]) * span
            dh += 0.5 * (p0.omega + p1.omega) * span
        return dx, dy, dh

from dataclasses import dataclass, field

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
    """Tagged point on the clip timeline.

    Markers are authored on the shared resource; per-character progress
    (which occurrences have already fired) lives on the player.
    ``payload`` is forwarded verbatim when the marker fires.
    """

    t: float
    name: str
    payload: object = None


class AnimationClip:
    """Immutable action resource shared by any number of character instances."""

    def __init__(self, name: str, keyframes, loop: bool, markers=()):
        frames = sorted(keyframes, key=lambda f: f.t)
        if len(frames) < 2:
            raise ValueError(f"clip {name!r} needs at least two keyframes")
        if frames[0].t != 0.0:
            raise ValueError(f"clip {name!r} must start at t=0")
        self.name = name
        self.keyframes = frames
        self.duration = frames[-1].t
        self.loop = loop
        self.joint_count = len(frames[0].joints)
        ordered_markers = sorted(markers, key=lambda m: m.t)
        names = [m.name for m in ordered_markers]
        if len(set(names)) != len(names):
            raise ValueError(f"clip {name!r} marker names must be unique")
        for marker in ordered_markers:
            if not 0.0 <= marker.t <= self.duration:
                raise ValueError(
                    f"marker {marker.name!r} of clip {name!r} is outside [0, duration]"
                )
        self.markers = tuple(ordered_markers)

    def marker(self, name: str) -> Marker:
        for marker in self.markers:
            if marker.name == name:
                return marker
        raise KeyError(f"clip {self.name!r} has no marker {name!r}")

    def marker_phase(self, marker_name: str, current_time: float) -> float:
        """Normalized position after the most recent marker occurrence."""
        marker = self.marker(marker_name)
        if self.loop:
            return ((current_time - marker.t) % self.duration) / self.duration
        if current_time <= marker.t:
            return 0.0
        if self.duration == marker.t:
            return 1.0
        return min(1.0, (current_time - marker.t) / (self.duration - marker.t))

    def sync_time(self, marker_name, source_clip=None, current_time=None):
        """Target time preserving progress through the named contact phase.

        Two looping clips can use different durations.  The fractional
        distance from the source marker is mapped onto the target cycle,
        rather than snapping the cursor onto a marker.  The result uses the
        positive phase representative, which gives the same distance back to
        the previous marker during reverse playback as forward playback gives
        to the next cyclic occurrence.
        """
        if isinstance(source_clip, (int, float)):
            current_time = source_clip
            source_clip = self
        if source_clip is None:
            source_clip = self
        if current_time is None:
            current_time = 0.0
        target_marker = self.marker(marker_name)
        if not self.loop or not getattr(source_clip, "loop", False):
            return target_marker.t
        phase = source_clip.marker_phase(marker_name, current_time)
        return (target_marker.t + phase * self.duration) % self.duration

    def crossed_markers(self, t0: float, t1: float):
        """Marker occurrences crossed while moving forward t0 -> t1.

        Half-open interval ``(t0, t1]``: arriving exactly on a marker fires
        it, but merely standing on one does not. Occurrences are returned in
        chronological order as ``(absolute_time, cycle_index, marker)``; the
        cycle index lets the same marker fire once per loop iteration.
        """
        if t1 <= t0:
            return []
        hits = []
        if self.loop:
            duration = self.duration
            for marker in self.markers:
                first = int((t0 - marker.t) // duration) + 1
                last = int((t1 - marker.t) // duration)
                for k in range(first, last + 1):
                    absolute = k * duration + marker.t
                    if t0 < absolute <= t1:
                        hits.append((absolute, k, marker))
        else:
            end = min(t1, self.duration)
            for marker in self.markers:
                if t0 < marker.t <= end:
                    hits.append((marker.t, 0, marker))
        hits.sort(key=lambda item: item[0])
        return hits

    def _normalise_time(self, t: float) -> float:
        if self.loop:
            return t % self.duration
        return clamp(t, 0.0, self.duration)

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

        Handles loop wraps by splitting at the wrap point and keyframe times,
        and supports signed ``dt`` (negative = reverse playback). Non-looping
        motion is clamped at the clip ends. Returns
        ``(dx, dy, d_heading)`` in the clip local frame.
        """
        if dt == 0.0:
            return 0.0, 0.0, 0.0
        sign = 1.0 if dt > 0.0 else -1.0
        low, high = sorted((t0, t0 + dt))
        if not self.loop:
            low = max(low, 0.0)
            high = min(high, self.duration)
            if high <= low:
                return 0.0, 0.0, 0.0
        breaks = {low, high}
        if self.loop:
            boundary = (int(low // self.duration) + 1) * self.duration
            while boundary < high:
                breaks.add(boundary)
                boundary += self.duration
        for frame in self.keyframes:
            frame_t = frame.t
            if self.loop:
                start_k = int(low // self.duration) - 1
                for k in range(start_k, int(high // self.duration) + 2):
                    candidate = k * self.duration + frame_t
                    if low < candidate < high:
                        breaks.add(candidate)
            elif low < frame_t < high:
                breaks.add(frame_t)

        dx = dy = dh = 0.0
        ordered = sorted(breaks)
        for left, right in zip(ordered, ordered[1:]):
            p0 = self.sample(left)
            p1 = self.sample(right)
            span = right - left
            dx += 0.5 * (p0.root_velocity[0] + p1.root_velocity[0]) * span
            dy += 0.5 * (p0.root_velocity[1] + p1.root_velocity[1]) * span
            dh += 0.5 * (p0.omega + p1.omega) * span
        return sign * dx, sign * dy, sign * dh

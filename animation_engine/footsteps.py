"""Per-foot contact constraints on uneven, editable terrain.

A :class:`FootstepTracker` rides on top of an :class:`AnimationPlayer`:
each update first advances the player, then derives per-foot contact
events (plant / lift) from the stride phase of the playing clip.  Feet in
their contact interval are held at the world position where they touched
down while the root keeps moving; a foot may only choose a new landing
spot after it has gone airborne again.

Design rules:

- Confirmed contacts are append-only history.  Changing the speed
  schedule, the target cycle duration or the clip only reschedules
  *future* plants; records already written are never rewritten, and
  later terrain edits do not retroactively change them either.
- A plant that cannot be satisfied (no ground, a segment the foot is not
  allowed on, or a step taller than max_step_height) is recorded as
  unsatisfied with the reason.  The character is never frozen to hide
  the problem: the root keeps advancing and the failed foot accumulates
  *slip* (the distance it is dragged) instead of a fake plant.
- Plant positions and times are pure functions of the clip-time interval
  advanced, so replaying the same span in many small update steps or
  one big step yields identical contact records.
- Reverse playback creates no contacts and removes none, mirroring the
  event rules of the player.
"""

import math
from dataclasses import dataclass

from .math_utils import clamp
from .player import AnimationPlayer

OK = "ok"
NO_GROUND = "no_ground"
SEGMENT_NOT_ALLOWED = "segment_not_allowed"
STEP_TOO_HIGH = "step_too_high"

_EPS = 1e-9


@dataclass(frozen=True)
class GroundSegment:
    """Flat walkable piece of ground along the world x axis."""

    id: str
    x0: float
    x1: float
    height: float

    def contains(self, x: float) -> bool:
        return self.x0 - _EPS <= x <= self.x1 + _EPS


class Terrain:
    """Piecewise-flat ground profile; deliberately editable between runs."""

    def __init__(self, segments):
        self.segments = list(segments)

    def segment_at(self, x: float):
        for segment in self.segments:
            if segment.contains(x):
                return segment
        return None

    def height_at(self, x: float):
        segment = self.segment_at(x)
        return None if segment is None else segment.height


@dataclass(frozen=True)
class FootSpec:
    """Contact constraint for one foot.

    contact_interval is the phase range (start, end) inside one stride
    cycle during which the foot must be planted. stride_offset is the
    forward offset from the root at which the foot wants to land.
    allowed_segments restricts where it may touch down (empty = any
    segment). max_step_height caps the height difference against the
    last confirmed plant; None disables the check.
    """

    name: str
    contact_interval: tuple
    stride_offset: float
    allowed_segments: tuple = ()
    max_step_height: float = None


def _new_foot_state():
    return {
        "status": "airborne",  # airborne | planted | unsatisfied
        "position": None,
        "height": None,
        "segment_id": None,
        "plant_time": None,
        "desired": None,
        "reason": None,
        "slip": 0.0,
        "record": None,
    }


class FootstepTracker:
    """Derives and records per-foot contacts from a player's stride phase."""

    def __init__(self, player, foot_specs, terrain, marker: str = "foot_down"):
        self.player = player
        self.terrain = terrain
        self.marker = marker
        self.specs = {spec.name: spec for spec in foot_specs}
        self.feet = {spec.name: _new_foot_state() for spec in foot_specs}
        self.clock = 0.0
        # Height reference for step-up checks: last confirmed plant.
        self.reference_height = None
        # Closed contact records; append-only history.
        self.records = []

    # -- phase helpers ---------------------------------------------------

    def _phase_window(self, clip):
        """Unwrapped stride phase at the current cursor, or None."""
        if not any(m.name == self.marker for m in clip.markers):
            return None
        marker = clip.marker(self.marker)
        period = clip.contact_period(self.marker)
        if period <= 0.0:
            return None
        return marker, period

    @staticmethod
    def _crossings(u0: float, u1: float, phase: float):
        """Phase-boundary crossings inside the half-open interval (u0, u1]."""
        k = math.floor(u0 - phase) + 1
        while True:
            u = k + phase
            if u > u1 + _EPS:
                return
            if u > u0 + _EPS:
                yield u
            k += 1

    def _wall_dt(self, clip, t_a: float, t_b: float) -> float:
        """Wall-clock seconds the player needs for clip span [t_a, t_b]."""
        st = self.player.state
        schedule = st.speed_schedule
        if schedule is None:
            return (t_b - t_a) / st.speed
        scale = schedule.time_scale()
        duration = clip.duration
        total = 0.0
        t = t_a
        while t < t_b - 1e-12:
            phase = t % duration if clip.loop else clamp(t, 0.0, duration)
            index = schedule.segment_index(phase)
            boundary = (t - phase) + schedule.segments[index].end
            step = min(t_b, boundary) - t
            total += step / (st.speed * schedule.segments[index].rate * scale)
            t += step
        return total

    # -- main entry ------------------------------------------------------

    def update(self, dt: float):
        """Advance the player by dt and derive contact events."""
        clip = self.player.clip
        t0 = self.player.state.local_time
        root0 = tuple(self.player.root_position)
        heading0 = self.player.root_heading
        clock0 = self.clock

        pose = self.player.update(dt)
        self.clock += dt

        t1 = self.player.state.local_time
        root1 = tuple(self.player.root_position)
        if t1 <= t0 + 1e-12:
            # Reverse playback or a clamped cursor: contacts are neither
            # created nor destroyed, exactly like marker events.
            return pose
        window = self._phase_window(clip)
        if window is None:
            return pose
        marker, period = window
        u0 = (t0 - marker.t) / period
        u1 = (t1 - marker.t) / period

        events = []
        for spec in self.specs.values():
            start, end = spec.contact_interval
            for u in self._crossings(u0, u1, start):
                events.append((u, spec, "plant"))
            for u in self._crossings(u0, u1, end):
                events.append((u, spec, "lift"))
        events.sort(key=lambda item: item[0])

        for u, spec, kind in events:
            t_cross = marker.t + u * period
            dx, dy, dh = clip.integrate_root(t0, t_cross - t0)
            wx, wy = AnimationPlayer._local_to_world_displacement(
                heading0, dx, dy, dh
            )
            position = (root0[0] + wx, root0[1] + wy)
            wall_time = clock0 + self._wall_dt(clip, t0, t_cross)
            if kind == "plant":
                self._plant(spec, position, heading0, wall_time)
            else:
                self._lift(spec, wall_time)

        # Unsatisfied feet are dragged along with the body instead of the
        # character being frozen: the drag distance is reported as slip.
        moved = math.hypot(root1[0] - root0[0], root1[1] - root0[1])
        if moved > 0.0:
            for foot in self.feet.values():
                if foot["status"] == "unsatisfied":
                    foot["slip"] += moved
                    if foot["record"] is not None:
                        foot["record"]["slip"] = foot["slip"]
        return pose

    # -- contact lifecycle -----------------------------------------------

    def _evaluate(self, spec, x: float, reference_height):
        segment = self.terrain.segment_at(x)
        if segment is None:
            return None, NO_GROUND
        if spec.allowed_segments and segment.id not in spec.allowed_segments:
            return segment, SEGMENT_NOT_ALLOWED
        if (
            spec.max_step_height is not None
            and reference_height is not None
            and abs(segment.height - reference_height) > spec.max_step_height
        ):
            return segment, STEP_TOO_HIGH
        return segment, OK

    def _plant(self, spec, root_at_crossing, heading, wall_time):
        foot = self.feet[spec.name]
        desired = (
            root_at_crossing[0] + spec.stride_offset * math.cos(heading),
            root_at_crossing[1] + spec.stride_offset * math.sin(heading),
        )
        segment, reason = self._evaluate(spec, desired[0], self.reference_height)
        record = {
            "foot": spec.name,
            "clip": self.player.state.clip_name,
            "plant_time": wall_time,
            "lift_time": None,
            "position": desired if reason == OK else None,
            "desired": desired,
            "height": segment.height if (segment and reason == OK) else None,
            "segment_id": segment.id if (segment and reason == OK) else None,
            "reason": reason,
            "slip": 0.0,
            "status": "active",
        }
        foot["record"] = record
        foot["plant_time"] = wall_time
        foot["desired"] = desired
        foot["reason"] = None if reason == OK else reason
        foot["slip"] = 0.0
        if reason == OK:
            foot["status"] = "planted"
            foot["position"] = desired
            foot["height"] = segment.height
            foot["segment_id"] = segment.id
            self.reference_height = segment.height
        else:
            # No fake plant: the foot stays unsatisfied and will slip.
            foot["status"] = "unsatisfied"
            foot["position"] = None
            foot["height"] = None
            foot["segment_id"] = None

    def _lift(self, spec, wall_time):
        foot = self.feet[spec.name]
        record = foot["record"]
        if record is not None:
            record["lift_time"] = wall_time
            record["slip"] = foot["slip"]
            record["status"] = (
                "satisfied" if record["reason"] == OK else "unsatisfied"
            )
            self.records.append(record)
        foot["status"] = "airborne"
        foot["position"] = None
        foot["height"] = None
        foot["segment_id"] = None
        foot["plant_time"] = None
        foot["desired"] = None
        foot["reason"] = None
        foot["slip"] = 0.0
        foot["record"] = None

    # -- inspection ------------------------------------------------------

    def contact_report(self):
        """Per-foot live state: status, actual contact position, slip, reason."""
        report = {}
        for name, foot in self.feet.items():
            report[name] = {
                "status": foot["status"],
                "position": foot["position"],
                "height": foot["height"],
                "segment_id": foot["segment_id"],
                "slip": foot["slip"],
                "reason": foot["reason"],
                "desired": foot["desired"],
            }
        return report

    def contact_history(self):
        """Closed contact records, oldest first; append-only."""
        return list(self.records)

    # -- planning --------------------------------------------------------

    def plan_steps(self, count: int, strategy: str = "keep",
                   adjust_limit: float = 0.5):
        """Simulate the next count plants without mutating any state.

        strategy="keep" replays the authored stride offsets;
        strategy="adjust" additionally tries scaled offsets (within
        +-adjust_limit) to reach an allowed, reachable segment.  Steps
        that still cannot be satisfied keep their unsatisfied status and
        reason, so the two strategies can be compared side by side.
        """
        clip = self.player.clip
        window = self._phase_window(clip)
        if window is None:
            return []
        marker, period = window
        t = self.player.state.local_time
        u_now = (t - marker.t) / period
        root = tuple(self.player.root_position)
        heading = self.player.root_heading
        clock = self.clock
        reference = self.reference_height

        events = []
        k0 = math.floor(u_now)
        for k in range(k0, k0 + count + 2):
            for spec in self.specs.values():
                u = k + spec.contact_interval[0]
                if u > u_now + _EPS:
                    events.append((u, spec))
        events.sort(key=lambda item: item[0])

        if strategy == "adjust":
            steps = int(round(adjust_limit / 0.05))
            scales = [1.0]
            for i in range(1, steps + 1):
                scales.append(1.0 + 0.05 * i)
                scales.append(1.0 - 0.05 * i)
        else:
            scales = [1.0]

        plan = []
        for u, spec in events[:count]:
            t_cross = marker.t + u * period
            dx, dy, dh = clip.integrate_root(t, t_cross - t)
            wx, wy = AnimationPlayer._local_to_world_displacement(
                heading, dx, dy, dh
            )
            root = (root[0] + wx, root[1] + wy)
            heading += dh
            clock += self._wall_dt(clip, t, t_cross)
            t = t_cross

            chosen = None
            for scale in scales:
                offset = spec.stride_offset * scale
                x = root[0] + offset * math.cos(heading)
                segment, reason = self._evaluate(spec, x, reference)
                if reason == OK:
                    chosen = (offset, x, segment)
                    break
            if chosen is None:
                offset = spec.stride_offset
                x = root[0] + offset * math.cos(heading)
                segment, reason = self._evaluate(spec, x, reference)
                plan.append({
                    "foot": spec.name,
                    "time": clock,
                    "position": (x, root[1] + offset * math.sin(heading)),
                    "stride_offset": offset,
                    "status": "unsatisfied",
                    "reason": reason,
                    "segment_id": segment.id if segment else None,
                })
            else:
                offset, x, segment = chosen
                reference = segment.height
                plan.append({
                    "foot": spec.name,
                    "time": clock,
                    "position": (x, root[1] + offset * math.sin(heading)),
                    "stride_offset": offset,
                    "status": "satisfied",
                    "reason": OK,
                    "segment_id": segment.id,
                })
        return plan

    def compare_plans(self, count: int, adjust_limit: float = 0.5):
        """Side-by-side 'keep original gait' vs 'adjust upcoming strides'."""
        return {
            "keep": self.plan_steps(count, "keep"),
            "adjust": self.plan_steps(count, "adjust", adjust_limit),
        }

    # -- persistence -----------------------------------------------------

    def save_state(self) -> dict:
        return {
            "clock": self.clock,
            "reference_height": self.reference_height,
            "marker": self.marker,
            "feet": {name: dict(foot) for name, foot in self.feet.items()},
            "records": [dict(record) for record in self.records],
        }

    def load_state(self, data: dict):
        self.clock = data["clock"]
        self.reference_height = data.get("reference_height")
        self.marker = data.get("marker", self.marker)
        for name, foot in data["feet"].items():
            if name in self.feet:
                restored = dict(foot)
                if restored["record"] is not None:
                    restored["record"] = dict(restored["record"])
                self.feet[name] = restored
        self.records = [dict(record) for record in data.get("records", [])]


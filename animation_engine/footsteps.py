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
    """Piecewise-flat ground profile; deliberately editable between runs.

    Beyond direct edits of the base segments list, terrain knowledge can
    grow through *registered observations*: each observation is a
    numbered, retract-able revision that adds segments (filling gaps) or
    replaces same-id segments (revising heights).  Observations apply on
    top of the base layer in registration order, so every version of the
    terrain stays reproducible and a footfall record can name the exact
    version it was evaluated against.
    """

    def __init__(self, segments):
        self.segments = list(segments)
        # Registered observations; a retracted entry becomes None so the
        # remaining version numbers stay stable for records citing them.
        self._observations = []

    @property
    def current_version(self) -> int:
        """Version a new evaluation would be stamped with (0 = base only)."""
        return len(self._observations)

    def register_observation(self, segments, note: str = None,
                             batch: str = None) -> int:
        """Commit a surveyed revision; returns its 1-based version id.

        ``batch`` tags the observation as part of a named survey batch.
        Conflicting batches coexist: the latest registration wins in the
        default (fully stacked) view, but each batch stays selectable on
        its own via ``snapshot(batch=...)`` / ``height_at(x, batch=...)``.
        """
        self._observations.append({
            "segments": list(segments),
            "note": note,
            "batch": batch,
        })
        return len(self._observations)

    def retract_observation(self, version: int):
        """Withdraw a previously registered observation."""
        if not 1 <= version <= len(self._observations):
            raise ValueError(f"unknown observation version {version}")
        if self._observations[version - 1] is None:
            raise ValueError(f"observation {version} already retracted")
        self._observations[version - 1] = None

    def batch_versions(self, batch: str):
        """1-based version ids of all live observations in a batch."""
        return [
            index + 1
            for index, entry in enumerate(self._observations)
            if entry is not None and entry.get("batch") == batch
        ]

    def snapshot(self, version: int = None, batch: str = None):
        """Resolved segment list at the given version (default: latest).

        With ``batch`` set, only that batch's observations are layered on
        the base segments, so one of several conflicting surveys can be
        selected without the others leaking into the evaluation.
        """
        if version is None:
            version = len(self._observations)
        resolved = {segment.id: segment for segment in self.segments}
        order = [segment.id for segment in self.segments]
        for entry in self._observations[:version]:
            if entry is None:
                continue
            if batch is not None and entry.get("batch") != batch:
                continue
            for segment in entry["segments"]:
                if segment.id not in resolved:
                    order.append(segment.id)
                resolved[segment.id] = segment
        return [resolved[segment_id] for segment_id in order]

    def segment_at(self, x: float, version: int = None, batch: str = None):
        for segment in self.snapshot(version, batch=batch):
            if segment.contains(x):
                return segment
        return None

    def height_at(self, x: float, version: int = None, batch: str = None):
        segment = self.segment_at(x, version, batch=batch)
        return None if segment is None else segment.height

    def save_state(self) -> dict:
        return {
            "segments": list(self.segments),
            "observations": [
                None if entry is None else dict(entry)
                for entry in self._observations
            ],
        }

    def load_state(self, data: dict):
        self.segments = list(data["segments"])
        self._observations = [
            None if entry is None else dict(entry)
            for entry in data.get("observations", [])
        ]


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
        # Watermark (in unwrapped stride-phase units) up to which contact
        # events have already been derived for the current clip.  Rewind
        # or seek-back followed by forward replay must not re-derive
        # crossings at or below it, mirroring the player event rules.
        self._derived_until = None
        self._derived_clip = None

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
        clip_name = self.player.state.clip_name
        if self._derived_clip != clip_name:
            self._derived_clip = clip_name
            self._derived_until = None
        if self._derived_until is not None:
            u0 = max(u0, self._derived_until)
        self._derived_until = u1

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

    def _evaluate(self, spec, x: float, reference_height, version=None,
                  batch=None):
        segment = self.terrain.segment_at(x, version, batch=batch)
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
            # Basis the evaluation was made against: the terrain version
            # and the reference height in force at plant time.  Later
            # observations never rewrite these fields.
            "terrain_version": self.terrain.current_version,
            "reference_height": self.reference_height,
            # Re-interpretations registered afterwards, newest last.
            "interpretations": [],
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

    # -- reinterpretation ------------------------------------------------

    @staticmethod
    def effective_record(record):
        """Current explanation of a record: latest interpretation applied.

        The original fields (the basis at plant time) are never touched;
        the view adds them under basis_* keys so both the original
        no_ground and the new explanation stay queryable.
        """
        view = dict(record)
        view["basis_reason"] = record["reason"]
        view["basis_terrain_version"] = record.get("terrain_version")
        view["batch"] = None
        if record.get("interpretations"):
            view.update(record["interpretations"][-1])
        return view

    def interpreted_history(self):
        """Closed records under their current (latest) explanation."""
        return [self.effective_record(record) for record in self.records]

    def reinterpret(self, start_time: float, end_time: float,
                    version: int = None, batch: str = None):
        """Re-evaluate unsatisfied plants in a wall-clock span of history
        against a (possibly newer) terrain version or a named survey batch.

        Only unsatisfied records are eligible: confirmed supports are
        frozen and are never silently moved - they are counted in the
        result instead.  The original record keeps its basis; the new
        outcome is appended to its interpretations list together with the
        source (version and batch) it was derived from.  Positions still
        without any observation stay pending (unsatisfied / no_ground).
        """
        if version is not None and batch is not None:
            raise ValueError("pass either version or batch, not both")
        if batch is not None:
            versions = self.terrain.batch_versions(batch)
            if not versions:
                raise ValueError("unknown observation batch %r" % (batch,))
            version = versions[-1]
        if version is None:
            version = self.terrain.current_version
        reinterpreted = 0
        frozen = 0
        for record in self.records:
            plant_time = record["plant_time"]
            if not (start_time - _EPS <= plant_time <= end_time + _EPS):
                continue
            if record["reason"] == OK:
                frozen += 1
                continue
            spec = self.specs[record["foot"]]
            segment, reason = self._evaluate(
                spec, record["desired"][0], record["reference_height"],
                version=None if batch is not None else version,
                batch=batch,
            )
            record["interpretations"].append({
                "terrain_version": version,
                "batch": batch,
                "reason": reason,
                "status": "satisfied" if reason == OK else "unsatisfied",
                "position": record["desired"] if reason == OK else None,
                "height": segment.height if (segment and reason == OK) else None,
                "segment_id": segment.id if (segment and reason == OK) else None,
            })
            reinterpreted += 1
        return {
            "reinterpreted": reinterpreted,
            "frozen": frozen,
            "terrain_version": version,
        }

    def retract_observation(self, version: int):
        """Withdraw a terrain observation and restore pending records.

        Interpretations that were based on the retracted version are
        dropped, so the affected records fall back to their previous
        explanation (or to the original pending/unsatisfied basis).
        Records and feet unrelated to that version are untouched.
        """
        self.terrain.retract_observation(version)
        restored = 0
        for record in self.records:
            kept = [
                entry for entry in record["interpretations"]
                if entry["terrain_version"] != version
            ]
            if len(kept) != len(record["interpretations"]):
                record["interpretations"] = kept
                restored += 1
        return restored

    def retract_batch(self, batch: str):
        """Withdraw a whole survey batch and restore dependent records.

        Every live observation of the batch is retracted and only the
        interpretations that were derived from this batch are dropped;
        explanations sourced from other batches and the original bases
        stay in place.  Returns the number of records restored.
        """
        versions = self.terrain.batch_versions(batch)
        if not versions:
            raise ValueError("unknown observation batch %r" % (batch,))
        for version in versions:
            self.terrain.retract_observation(version)
        restored = 0
        for record in self.records:
            kept = [
                entry for entry in record["interpretations"]
                if entry.get("batch") != batch
                and entry["terrain_version"] not in versions
            ]
            if len(kept) != len(record["interpretations"]):
                record["interpretations"] = kept
                restored += 1
        return restored

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
        terrain = getattr(self.terrain, "save_state", None)
        return {
            "clock": self.clock,
            "reference_height": self.reference_height,
            "marker": self.marker,
            "derived_until": self._derived_until,
            "derived_clip": self._derived_clip,
            "feet": {name: dict(foot) for name, foot in self.feet.items()},
            "records": [self._copy_record(record) for record in self.records],
            "terrain": terrain() if callable(terrain) else None,
        }

    def load_state(self, data: dict):
        self.clock = data["clock"]
        self.reference_height = data.get("reference_height")
        self.marker = data.get("marker", self.marker)
        self._derived_until = data.get("derived_until")
        self._derived_clip = data.get("derived_clip")
        for name, foot in data["feet"].items():
            if name in self.feet:
                restored = dict(foot)
                if restored["record"] is not None:
                    restored["record"] = self._copy_record(restored["record"])
                self.feet[name] = restored
        self.records = [
            self._copy_record(record) for record in data.get("records", [])
        ]
        terrain_data = data.get("terrain")
        if terrain_data is not None and hasattr(self.terrain, "load_state"):
            self.terrain.load_state(terrain_data)

    @staticmethod
    def _copy_record(record):
        copied = dict(record)
        copied["interpretations"] = [
            dict(entry) for entry in record.get("interpretations", [])
        ]
        return copied

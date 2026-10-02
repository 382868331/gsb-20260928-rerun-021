"""Foot-plant constraints on uneven ground, per character.

A :class:`GaitTracker` wraps an :class:`AnimationPlayer` and gives each
foot an independent, explainable contact state:

- The user authors, per foot, a contact window (phase interval of the
  clip cycle), a nominal offset from the root, the ground segments the
  foot may land on and an optional reach limit.
- While a foot is inside its contact window it stays glued to the
  ground position it confirmed at plant time; the root keeps moving and
  the difference is reported as *slide* instead of being hidden by
  locking the whole character.
- Future plant times follow the live playback arrangement: changing
  speed, installing a speed schedule or switching clips only moves
  plants that have not happened yet. Confirmed contacts are stored as
  plain values and never recomputed, so later terrain edits cannot
  rewrite history.
- Reverse playback rewinds pose and root only; feet keep their current
  state and no plants or liftoffs are produced while rewinding.

The tracker sub-steps the player so every contact window boundary is
landed on exactly. Because boundaries are hit at their exact phase
regardless of how the caller chunks ``dt``, chunked updates and a
single whole-span update produce identical contact records.
"""

import math

from .math_utils import clamp

_EPS = 1e-9
_PHASE_TOL = 1e-6


class GroundSegment:
    """Flat walkable piece of ground ``[x0, x1]`` at a constant height."""

    def __init__(self, x0: float, x1: float, height: float = 0.0,
                 name: str = None):
        if not x1 > x0:
            raise ValueError("ground segment needs x1 > x0")
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.height = float(height)
        self.name = name

    def contains(self, x: float) -> bool:
        return self.x0 - _EPS <= x <= self.x1 + _EPS

    def __repr__(self):
        return f"GroundSegment({self.x0}, {self.x1}, {self.height!r})"


class Terrain:
    """Editable set of ground segments; plants reference it by index."""

    def __init__(self, segments=()):
        self.segments = list(segments)

    def add_segment(self, segment: GroundSegment) -> int:
        self.segments.append(segment)
        return len(self.segments) - 1

    def segment_at(self, x: float):
        """Index of the segment covering ``x``, or ``None`` over a gap."""
        for index, segment in enumerate(self.segments):
            if segment.contains(x):
                return index
        return None

    def height_at(self, x: float):
        index = self.segment_at(x)
        return None if index is None else self.segments[index].height

    def nearest_reachable(self, x: float, root_x: float, max_reach,
                          allowed):
        """Point nearest to ``x`` that satisfies reach and allow-list.

        Returns ``(x, segment_index)`` or ``None`` when no allowed
        segment intersects the reachable interval.
        """
        if max_reach is None:
            low, high = -math.inf, math.inf
        else:
            low, high = root_x - max_reach, root_x + max_reach
        best = None
        for index, segment in enumerate(self.segments):
            if allowed is not None and index not in allowed:
                continue
            lo = max(segment.x0, low)
            hi = min(segment.x1, high)
            if lo > hi + _EPS:
                continue
            candidate = min(max(x, lo), hi)
            distance = abs(candidate - x)
            if best is None or distance < best[0]:
                best = (distance, candidate, index)
        if best is None:
            return None
        return best[1], best[2]


class FootSpec:
    """Authored per-foot constraint.

    ``window`` is ``(start_phase, end_phase)`` of the clip cycle during
    which the foot should be on the ground; ``start > end`` wraps around
    the cycle boundary. ``offset`` is the nominal foot position in the
    root frame. ``allowed`` restricts plantable segment indices (``None``
    = any). ``max_reach`` caps the horizontal distance from the root.
    """

    def __init__(self, name: str, window, offset=(0.0, 0.0),
                 allowed=None, max_reach=None):
        start, end = window
        if not (0.0 <= start < 1.0 and 0.0 < end <= 1.0):
            raise ValueError("contact window phases must lie in [0, 1]")
        if start == end:
            raise ValueError("contact window must have non-zero width")
        self.name = name
        self.window = (float(start), float(end))
        self.offset = (float(offset[0]), float(offset[1]))
        self.allowed = None if allowed is None else set(allowed)
        self.max_reach = max_reach


class GaitTracker:
    """Drives a player and maintains per-foot contact state on terrain."""

    def __init__(self, player, terrain: Terrain, feet):
        self.player = player
        self.terrain = terrain
        self.feet = {foot.name: foot for foot in feet}
        self.wall_clock = 0.0
        self._states = {
            name: self._airborne_state(name) for name in self.feet
        }
        self._open_records = {}
        self._records = []
        self._events = []
        self._primed = False

    @staticmethod
    def _airborne_state(name):
        return {
            "foot": name,
            "status": "airborne",
            "position": None,
            "height": None,
            "segment": None,
            "since": None,
            "slide": 0.0,
            "reason": None,
        }

    # -- phase / timing -------------------------------------------------

    def _phase(self) -> float:
        st = self.player.state
        clip = self.player.clip
        if clip.duration <= 0.0:
            return 0.0
        if clip.loop:
            return (st.local_time % clip.duration) / clip.duration
        return clamp(st.local_time, 0.0, clip.duration) / clip.duration

    def _forward_clip_distance(self, target_phase: float):
        clip = self.player.clip
        duration = clip.duration
        current = self._phase() * duration
        target = target_phase * duration
        if clip.loop:
            distance = (target - current) % duration
            if distance <= _EPS:
                distance = duration
            return distance
        distance = target - current
        return distance if distance > _EPS else None

    def _wall_dt_for(self, clip_distance: float) -> float:
        st = self.player.state
        if st.speed <= 0.0:
            return math.inf
        schedule = st.speed_schedule
        if schedule is None:
            return clip_distance / st.speed
        clip = self.player.clip
        duration = clip.duration
        scale = schedule.time_scale()
        if clip.loop:
            phase_t = st.local_time % duration
        else:
            phase_t = clamp(st.local_time, 0.0, duration)
        remaining = clip_distance
        wall = 0.0
        for _ in range(4096):
            index = schedule.segment_index(phase_t % duration)
            segment = schedule.segments[index]
            piece = min(remaining, segment.end - phase_t)
            wall += piece / (st.speed * segment.rate * scale)
            remaining -= piece
            if remaining <= _EPS:
                return wall
            phase_t = segment.end
            if phase_t >= duration - _EPS:
                phase_t = 0.0
        return math.inf

    def _next_boundary_dt(self):
        """Smallest wall-clock dt to any contact window boundary."""
        best = math.inf
        for foot in self.feet.values():
            for boundary in foot.window:
                distance = self._forward_clip_distance(boundary)
                if distance is None:
                    continue
                best = min(best, self._wall_dt_for(distance))
        return best

    # -- contact transitions --------------------------------------------

    def _animated_position(self, foot: FootSpec):
        st = self.player.state
        cos_h = math.cos(st.root_heading)
        sin_h = math.sin(st.root_heading)
        ox, oy = foot.offset
        return (
            st.root_position[0] + ox * cos_h - oy * sin_h,
            st.root_position[1] + ox * sin_h + oy * cos_h,
        )

    def _validate_plant(self, foot: FootSpec, x: float):
        """``(segment_index, reason)``; reason is ``None`` when valid."""
        st = self.player.state
        index = self.terrain.segment_at(x)
        if index is None:
            return None, "no_ground"
        if foot.allowed is not None and index not in foot.allowed:
            return None, "segment_not_allowed"
        if (
            foot.max_reach is not None
            and abs(x - st.root_position[0]) > foot.max_reach + _EPS
        ):
            return None, "out_of_reach"
        return index, None

    def _plant(self, foot: FootSpec):
        x, _ = self._animated_position(foot)
        index, reason = self._validate_plant(foot, x)
        state = self._states[foot.name]
        if reason is None:
            height = self.terrain.segments[index].height
            state.update(
                status="planted",
                position=(x, height),
                height=height,
                segment=index,
                since=self.wall_clock,
                slide=0.0,
                reason=None,
            )
            record = {
                "foot": foot.name,
                "plant_time": self.wall_clock,
                "lift_time": None,
                "position": (x, height),
                "segment": index,
                "satisfied": True,
                "reason": None,
                "max_slide": 0.0,
            }
            self._events.append({
                "type": "plant", "foot": foot.name,
                "time": self.wall_clock, "position": (x, height),
            })
        else:
            state.update(
                status="unsatisfied",
                position=None,
                height=None,
                segment=None,
                since=self.wall_clock,
                slide=0.0,
                reason=reason,
            )
            record = {
                "foot": foot.name,
                "plant_time": self.wall_clock,
                "lift_time": None,
                "position": None,
                "segment": None,
                "satisfied": False,
                "reason": reason,
                "max_slide": 0.0,
            }
            self._events.append({
                "type": "unsatisfied", "foot": foot.name,
                "time": self.wall_clock, "reason": reason,
            })
        self._open_records[foot.name] = record

    def _liftoff(self, foot: FootSpec):
        state = self._states[foot.name]
        if state["status"] == "airborne":
            return
        record = self._open_records.pop(foot.name, None)
        if record is not None:
            record["lift_time"] = self.wall_clock
            record["max_slide"] = max(record["max_slide"], state["slide"])
            self._records.append(record)
        self._events.append({
            "type": "liftoff", "foot": foot.name,
            "time": self.wall_clock,
            "satisfied": state["status"] == "planted",
        })
        self._states[foot.name] = self._airborne_state(foot.name)

    def _inside_window(self, foot: FootSpec, phase: float) -> bool:
        start, end = foot.window
        if start < end:
            return start - _PHASE_TOL <= phase < end - _PHASE_TOL
        return phase >= start - _PHASE_TOL or phase < end - _PHASE_TOL

    def _prime(self):
        """Plant feet whose window already covers the starting phase."""
        phase = self._phase()
        for foot in self.feet.values():
            if foot.name in self._open_records:
                continue
            if self._states[foot.name]["status"] != "airborne":
                continue
            if self._inside_window(foot, phase):
                self._plant(foot)
        self._primed = True

    def _process_boundaries(self):
        phase = self._phase()
        # Exits before entries, so at a shared boundary the old contact
        # closes before the next one opens.
        for foot in self.feet.values():
            end = foot.window[1]
            if (
                abs(phase - end) < _PHASE_TOL
                or abs(phase - end - 1.0) < _PHASE_TOL
                or abs(phase - end + 1.0) < _PHASE_TOL
            ):
                self._liftoff(foot)
        for foot in self.feet.values():
            start = foot.window[0]
            if (
                abs(phase - start) < _PHASE_TOL
                or abs(phase - start - 1.0) < _PHASE_TOL
                or abs(phase - start + 1.0) < _PHASE_TOL
            ):
                self._plant(foot)

    def _update_slides(self):
        for name, state in self._states.items():
            if state["status"] != "planted":
                continue
            ax, ay = self._animated_position(self.feet[name])
            px, py = state["position"]
            state["slide"] = math.hypot(ax - px, ay - py)

    # -- public API -----------------------------------------------------

    def update(self, dt: float):
        """Advance the player and reconcile foot contacts.

        Negative ``dt`` rewinds pose and root only: feet keep their
        current state and no plant/liftoff events are produced.
        """
        if dt < 0.0:
            pose = self.player.update(dt)
            self.wall_clock += dt
            return pose
        if dt == 0.0:
            return self.player.current_pose()
        if not self._primed:
            self._prime()
        pose = self.player.current_pose()
        remaining = dt
        while remaining > _EPS:
            boundary = self._next_boundary_dt()
            step = boundary
            if step > remaining:
                step = remaining
            step = max(step, 0.0)
            pose = self.player.update(step)
            self.wall_clock += step
            remaining -= step
            if boundary <= step + _EPS:
                self._process_boundaries()
            self._update_slides()
        return pose

    def contacts(self):
        """Snapshot of each foot's current contact state."""
        return {name: dict(state) for name, state in self._states.items()}

    def contact_records(self):
        """Closed contact records, oldest first. Never rewritten."""
        return [dict(record) for record in self._records]

    def pop_foot_events(self):
        events = self._events
        self._events = []
        return events

    def evaluate_plans(self, foot_name: str):
        """Compare plant options for a foot without committing anything.

        Returns one entry per strategy: ``keep_gait`` plants at the
        nominal animated position (reporting why it fails, if it does);
        ``adjust_stride`` shifts the plant to the nearest reachable
        allowed point, which shortens or lengthens the following stride.
        Unreachable goals stay unsatisfied in both.
        """
        foot = self.feet[foot_name]
        root_x = self.player.state.root_position[0]
        nominal_x, _ = self._animated_position(foot)
        index, reason = self._validate_plant(foot, nominal_x)
        plans = [{
            "strategy": "keep_gait",
            "position": (
                None
                if reason
                else (nominal_x, self.terrain.segments[index].height)
            ),
            "satisfied": reason is None,
            "reason": reason,
            "stride_adjustment": 0.0,
        }]
        found = self.terrain.nearest_reachable(
            nominal_x, root_x, foot.max_reach, foot.allowed
        )
        if found is None:
            plans.append({
                "strategy": "adjust_stride",
                "position": None,
                "satisfied": False,
                "reason": "no_reachable_ground",
                "stride_adjustment": None,
            })
        else:
            x, index = found
            plans.append({
                "strategy": "adjust_stride",
                "position": (x, self.terrain.segments[index].height),
                "satisfied": True,
                "reason": None,
                "stride_adjustment": x - nominal_x,
            })
        return plans

    # -- persistence ----------------------------------------------------

    def save_state(self) -> dict:
        return {
            "wall_clock": self.wall_clock,
            "states": {
                name: dict(state) for name, state in self._states.items()
            },
            "open_records": {
                name: dict(record)
                for name, record in self._open_records.items()
            },
            "records": [dict(record) for record in self._records],
            "events": list(self._events),
            "player": self.player.save_state(),
        }

    def load_state(self, data: dict):
        def restore_position(mapping):
            if mapping.get("position") is not None:
                mapping["position"] = tuple(mapping["position"])

        self.wall_clock = data["wall_clock"]
        self._states = {
            name: dict(state) for name, state in data["states"].items()
        }
        for state in self._states.values():
            restore_position(state)
        self._open_records = {
            name: dict(record)
            for name, record in data["open_records"].items()
        }
        for record in self._open_records.values():
            restore_position(record)
        self._records = [dict(record) for record in data["records"]]
        for record in self._records:
            restore_position(record)
        self._events = list(data["events"])
        self.player.load_state(data["player"])

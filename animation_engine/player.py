import math
from dataclasses import dataclass

from .math_utils import clamp, lerp, smoothstep
from .pose import Pose


def _canonical_loop_time(clip, local_time: float) -> float:
    if clip.loop:
        return local_time % clip.duration
    return clamp(local_time, 0.0, clip.duration)


@dataclass
class PlaybackState:
    clip_name: str
    local_time: float = 0.0
    speed: float = 1.0
    root_position: tuple = (0.0, 0.0)
    root_heading: float = 0.0
    blend_elapsed: float = 0.0
    blend_duration: float = 0.0
    frozen_joints: tuple = None
    frozen_velocity: tuple = None
    frozen_omega: float = 0.0
    # Bumped on every (re)entry into a clip; together with the marker name
    # and loop-cycle index it uniquely identifies an event occurrence.
    generation: int = 0
    # Occurrence keys that have already fired during this play-through.
    fired_events: tuple = ()

    @property
    def is_blending(self) -> bool:
        return self.blend_duration > 0.0 and self.blend_elapsed < self.blend_duration


class AnimationPlayer:
    """Per-character mutable playback cursor over immutable clip resources."""

    def __init__(self, library, clip_name: str, root_position=(0.0, 0.0)):
        self.library = library
        self.state = PlaybackState(
            clip_name=clip_name, root_position=tuple(root_position)
        )
        self._pending_events = []
        self._contact_records = []

    @property
    def clip(self):
        return self.library[self.state.clip_name]

    @property
    def root_position(self) -> tuple:
        return self.state.root_position

    @property
    def root_heading(self) -> float:
        return self.state.root_heading

    def current_pose(self) -> Pose:
        st = self.state
        target = self.clip.sample(st.local_time)
        if not st.is_blending:
            return target
        weight = smoothstep(st.blend_elapsed / st.blend_duration)
        joints = tuple(
            lerp(old, new, weight)
            for old, new in zip(st.frozen_joints, target.joints)
        )
        velocity = tuple(
            lerp(old, new, weight)
            for old, new in zip(st.frozen_velocity, target.root_velocity)
        )
        return Pose(joints, velocity, lerp(st.frozen_omega, target.omega, weight))

    def transition_to(self, clip_name: str, duration: float,
                      entry_time: float = 0.0, sync_marker: str = None):
        if clip_name == self.state.clip_name and not self.state.is_blending:
            return
        source_clip = self.clip
        source_time = self.state.local_time
        snapshot = self.current_pose()
        target_clip = self.library[clip_name]
        source_contact = None
        target_contact = None
        if sync_marker is not None:
            entry_time = target_clip.sync_time(
                sync_marker, source_clip, self.state.local_time
            )
            source_contact = self._contact_state(
                source_clip, sync_marker, self.state.local_time
            )
            target_contact = self._contact_state(
                target_clip, sync_marker, entry_time
            )
        self.state.clip_name = clip_name
        self.state.local_time = entry_time
        self.state.blend_elapsed = 0.0
        self.state.blend_duration = max(0.0, duration)
        self.state.frozen_joints = snapshot.joints
        self.state.frozen_velocity = snapshot.root_velocity
        self.state.frozen_omega = snapshot.omega
        # Switching clips starts a fresh play-through for the new clip. The
        # old clip's not-yet-reached markers are cancelled rather than
        # replayed through the target cursor; already queued events survive.
        self.state.generation += 1
        self.state.fired_events = ()
        self._contact_records.append(
            {
                "index": len(self._contact_records),
                "source_clip": source_clip.name,
                "target_clip": target_clip.name,
                "marker": sync_marker,
                "source_local_time": _canonical_loop_time(source_clip, source_time),
                "target_local_time": entry_time,
                "before": source_contact,
                "after": target_contact,
            }
        )

    def _contact_state(self, clip, marker_name: str, local_time: float):
        marker = clip.marker(marker_name)
        phase = clip.marker_phase(marker_name, local_time)
        return {
            "clip": clip.name,
            "local_time": _canonical_loop_time(clip, local_time),
            "marker": marker_name,
            "phase": phase,
            "payload": marker.payload,
            "grounded": math.isclose(phase, 0.0, abs_tol=1e-12),
        }

    def contact_records(self):
        """Read-only replay log of contact state around each transition."""
        return [dict(record) for record in self._contact_records]

    def seek(self, local_time: float):
        """Jump the cursor without blending or emitting events.

        Markers inside the skipped region are treated as neither fired nor
        pending: a jump never synthesises events. Fired occurrences stay
        fired, so seeking back and moving forward again does not replay
        them. Any in-progress blend is cancelled.
        """
        clip = self.clip
        st = self.state
        if not clip.loop:
            local_time = clamp(local_time, 0.0, clip.duration)
        st.local_time = local_time
        st.blend_elapsed = 0.0
        st.blend_duration = 0.0
        st.frozen_joints = None
        st.frozen_velocity = None
        st.frozen_omega = 0.0

    def queue_event(self, name: str, payload, cycle: int):
        st = self.state
        key = (st.generation, name, cycle)
        fired = set(st.fired_events)
        if key in fired:
            return
        fired.add(key)
        st.fired_events = tuple(sorted(fired))
        self._pending_events.append(
            {
                "clip": st.clip_name,
                "marker": name,
                "cycle": cycle,
                "payload": payload,
            }
        )

    def pop_events(self):
        """Return and clear external events produced since the last call."""
        events = self._pending_events
        self._pending_events = []
        return events

    def _collect_events(self, clip, t0: float, t1: float):
        for _, cycle, marker in clip.crossed_markers(t0, t1):
            self.queue_event(marker.name, marker.payload, cycle)

    def _advance_without_blend(self, clip, effective_dt: float):
        st = self.state
        start = st.local_time
        dx, dy, dh = clip.integrate_root(start, effective_dt)
        end = start + effective_dt
        if effective_dt > 0.0:
            self._collect_events(clip, start, end)
        # Reverse playback never fires events; moving backwards only rewinds
        # pose and root channels.
        if not clip.loop:
            end = clamp(end, 0.0, clip.duration)
        st.local_time = end
        return dx, dy, dh

    def _keyframe_breaks(self, clip, t0: float, dt: float):
        end = t0 + dt
        breaks = {0.0}
        for frame in clip.keyframes:
            marker = frame.t
            if clip.loop:
                first = t0 + ((marker - t0) % clip.duration)
                boundary = first
                while boundary < end - 1e-12:
                    if boundary > t0 + 1e-12:
                        breaks.add(boundary - t0)
                    boundary += clip.duration
            elif marker > 0.0 and t0 < marker <= end:
                breaks.add(marker - t0)
        return breaks

    def _frozen_weight_integral(self, start: float, end: float) -> float:
        st = self.state
        duration = st.blend_duration

        def primitive(u: float) -> float:
            x = u / duration
            return x - (x ** 3 - 0.5 * x ** 4)

        return duration * (primitive(end) - primitive(start))

    def _target_weight_integrals(self, start: float, end: float):
        st = self.state
        duration = st.blend_duration

        def weighted_time(u: float) -> float:
            return (
                0.75 * u ** 4 / duration ** 2
                - 0.4 * u ** 5 / duration ** 3
            )

        def weighted_area(u: float) -> float:
            x = u / duration
            return duration * (x ** 3 - 0.5 * x ** 4)

        area = weighted_area(end) - weighted_area(start)
        moment = weighted_time(end) - weighted_time(start)
        return area, moment

    def _integrate_blend(self, clip, t0: float, clip_dt: float,
                         wall_dt: float, frozen_velocity, frozen_omega):
        st = self.state
        breaks = sorted({0.0, clip_dt, *(
            value for value in self._keyframe_breaks(clip, t0, clip_dt)
            if 0.0 < value < clip_dt
        )})
        frozen_scale = self._frozen_weight_integral(
            st.blend_elapsed, st.blend_elapsed + wall_dt
        )
        result = [0.0, 0.0, 0.0]
        result[0] += frozen_velocity[0] * frozen_scale
        result[1] += frozen_velocity[1] * frozen_scale
        result[2] += frozen_omega * frozen_scale
        for clip_start, clip_end in zip(breaks, breaks[1:]):
            wall_start = clip_start / st.speed
            wall_end = clip_end / st.speed
            area, absolute_moment = self._target_weight_integrals(
                st.blend_elapsed + wall_start,
                st.blend_elapsed + wall_end,
            )
            relative_moment = (
                absolute_moment - (
                    st.blend_elapsed + wall_start
                ) * area
            )
            relative_moment *= st.speed
            pose0 = clip.sample(t0 + clip_start)
            pose1 = clip.sample(t0 + clip_end)
            values = (
                pose0.root_velocity[0],
                pose0.root_velocity[1],
                pose0.omega,
            )
            slopes = (
                (pose1.root_velocity[0] - pose0.root_velocity[0])
                / (clip_end - clip_start),
                (pose1.root_velocity[1] - pose0.root_velocity[1])
                / (clip_end - clip_start),
                (pose1.omega - pose0.omega)
                / (clip_end - clip_start),
            )
            for i in range(3):
                result[i] += values[i] * area + slopes[i] * relative_moment
        return tuple(result)

    def update(self, dt: float):
        if dt == 0.0 or dt * self.state.speed == 0.0:
            return self.current_pose()

        st = self.state
        effective_dt = dt * st.speed
        clip = self.clip
        dx = dy = dh = 0.0

        if effective_dt < 0.0 and st.is_blending:
            # Reverse is an explicit rule change: cancel the blend and move
            # the target clip backwards. No events fire while rewinding.
            st.blend_elapsed = 0.0
            st.blend_duration = 0.0
            st.frozen_joints = None
            st.frozen_velocity = None
            st.frozen_omega = 0.0

        if st.is_blending and effective_dt >= 0.0:
            wall_blend_dt = min(
                dt, st.blend_duration - st.blend_elapsed
            )
            active_clip_dt = wall_blend_dt * st.speed
            active_wall_dt = wall_blend_dt
            if not clip.loop:
                active_clip_dt = min(
                    active_clip_dt, clip.duration - st.local_time
                )
                active_wall_dt = active_clip_dt / st.speed
            move = self._integrate_blend(
                clip,
                st.local_time,
                active_clip_dt,
                active_wall_dt,
                st.frozen_velocity,
                st.frozen_omega,
            )
            dx += move[0]
            dy += move[1]
            dh += move[2]

            blend_end_time = st.local_time + active_clip_dt
            self._collect_events(clip, st.local_time, blend_end_time)
            st.local_time = blend_end_time
            st.blend_elapsed += active_wall_dt
            pose_only_dt = wall_blend_dt - active_wall_dt
            st.blend_elapsed += pose_only_dt
            if st.blend_elapsed >= st.blend_duration:
                st.blend_elapsed = 0.0
                st.blend_duration = 0.0
                st.frozen_joints = None
                st.frozen_velocity = None
                st.frozen_omega = 0.0
            tail_dt = effective_dt - wall_blend_dt * st.speed
            if tail_dt > 0.0:
                tail_start = st.local_time
                move = clip.integrate_root(tail_start, tail_dt)
                dx += move[0]
                dy += move[1]
                dh += move[2]
                tail_end = tail_start + tail_dt
                self._collect_events(clip, tail_start, tail_end)
                if not clip.loop:
                    tail_end = clamp(tail_end, 0.0, clip.duration)
                st.local_time = tail_end
        else:
            dx, dy, dh = self._advance_without_blend(clip, effective_dt)

        world_dx, world_dy = self._local_to_world_displacement(
            st.root_heading, dx, dy, dh
        )
        st.root_position = (
            st.root_position[0] + world_dx,
            st.root_position[1] + world_dy,
        )
        st.root_heading += dh
        return self.current_pose()

    @staticmethod
    def _local_to_world_displacement(heading, dx, dy, dh):
        if abs(dh) < 1e-12:
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h
        chord = 2.0 * math.sin(0.5 * dh) / dh
        average_heading = heading + 0.5 * dh
        cos_h = math.cos(average_heading) * chord
        sin_h = math.sin(average_heading) * chord
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    def save_state(self) -> dict:
        st = self.state
        return {
            "clip_name": st.clip_name,
            "local_time": st.local_time,
            "speed": st.speed,
            "root_position": st.root_position,
            "root_heading": st.root_heading,
            "blend_elapsed": st.blend_elapsed,
            "blend_duration": st.blend_duration,
            "frozen_joints": st.frozen_joints,
            "frozen_velocity": st.frozen_velocity,
            "frozen_omega": st.frozen_omega,
            "generation": st.generation,
            "fired_events": [list(key) for key in st.fired_events],
            "pending_events": list(self._pending_events),
            "contact_records": self.contact_records(),
        }

    def load_state(self, data: dict):
        self.state = PlaybackState(
            clip_name=data["clip_name"],
            local_time=data["local_time"],
            speed=data["speed"],
            root_position=tuple(data["root_position"]),
            root_heading=data.get("root_heading", 0.0),
            blend_elapsed=data["blend_elapsed"],
            blend_duration=data["blend_duration"],
            frozen_joints=data["frozen_joints"],
            frozen_velocity=data["frozen_velocity"],
            frozen_omega=data["frozen_omega"],
            generation=data.get("generation", 0),
            fired_events=tuple(
                tuple(key) for key in data.get("fired_events", ())
            ),
        )
        self._pending_events = list(data.get("pending_events", []))
        self._contact_records = [
            dict(record) for record in data.get("contact_records", [])
        ]

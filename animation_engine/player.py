import math
from dataclasses import dataclass

from .math_utils import lerp, smoothstep
from .math_utils import clamp
from .pose import Pose

_GAUSS_NODES = (
    0.046910077030668, 0.230765344947157, 0.5, 0.769234655052843,
    0.953089922969332,
)
_GAUSS_WEIGHTS = (
    0.118463442528095, 0.239314335249683, 0.284444444444444,
    0.239314335249683, 0.118463442528095,
)


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
        # One entry per requested transition: contact snapshots on both
        # sides, so users can replay the grounded-foot phase at each switch.
        self._transition_log = []

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

    @staticmethod
    def _contact_snapshot(clip, local_time: float, marker_name: str):
        """Grounded-foot phase: time elapsed past the marker occurrence.

        ``progress`` is the fraction of one stride (loop) or of the remaining
        clip span (one-shot); ``grounded`` covers the first half of that span,
        i.e. the stance phase following the foot-down instant.
        """
        marker = clip.marker(marker_name)
        offset = clip.contact_offset(marker_name, local_time)
        if clip.loop:
            period = clip.duration
            progress = (offset / period) % 1.0 if period else 0.0
        else:
            period = max(0.0, clip.duration - marker.t)
            progress = offset / period if period else 0.0
        return {
            "clip": clip.name,
            "local_time": local_time,
            "marker": marker_name,
            "payload": marker.payload,
            "offset": offset,
            "progress": progress,
            "grounded": offset <= 0.5 * period + 1e-12,
        }

    def contact_state(self, marker_name: str = "foot_down"):
        """Current grounded-foot phase of the playing clip."""
        return self._contact_snapshot(
            self.clip, self.state.local_time, marker_name
        )

    def transition_history(self):
        """Before/after contact snapshots for every transition requested."""
        return list(self._transition_log)

    def transition_to(self, clip_name: str, duration: float,
                      entry_time: float = 0.0, sync_marker: str = None,
                      new_speed: float = None):
        st = self.state
        if clip_name == st.clip_name and not st.is_blending:
            return
        source_clip = self.clip
        before_contact = (
            self._contact_snapshot(source_clip, st.local_time, sync_marker)
            if sync_marker is not None
            else None
        )
        speed_before = st.speed
        snapshot = self.current_pose()
        target_clip = self.library[clip_name]
        if sync_marker is not None:
            entry_time = target_clip.sync_time(
                sync_marker, source_clip, st.local_time
            )
        st.clip_name = clip_name
        st.local_time = entry_time
        st.blend_elapsed = 0.0
        st.blend_duration = max(0.0, duration)
        st.frozen_joints = snapshot.joints
        st.frozen_velocity = snapshot.root_velocity
        st.frozen_omega = snapshot.omega
        if new_speed is not None:
            st.speed = new_speed
        after_contact = (
            self._contact_snapshot(target_clip, entry_time, sync_marker)
            if sync_marker is not None
            else None
        )
        self._transition_log.append(
            {
                "source": source_clip.name,
                "destination": clip_name,
                "duration": st.blend_duration,
                "speed_before": speed_before,
                "speed_after": new_speed if new_speed is not None else speed_before,
                "before": before_contact,
                "after": after_contact,
            }
        )
        # Switching clips starts a fresh play-through for the new clip. The
        # old clip's not-yet-reached markers are cancelled rather than
        # replayed through the target cursor; already queued events survive.
        self.state.generation += 1
        self.state.fired_events = ()

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

    def _velocity_breaks(self, clip, t0: float, dt: float):
        end = t0 + dt
        breaks = {0.0, dt}
        for frame in clip.keyframes:
            marker = frame.t
            if clip.loop:
                first = ((t0 - marker) // clip.duration + 1) * clip.duration + marker
                boundary = first
                while boundary < end - 1e-12:
                    if boundary > t0 + 1e-12:
                        breaks.add(boundary - t0)
                    boundary += clip.duration
            elif t0 < marker < end:
                breaks.add(marker - t0)
        blend_end = self.state.blend_duration - self.state.blend_elapsed
        if 0.0 < blend_end < dt:
            breaks.add(blend_end)
        return sorted(breaks)

    def _frozen_weight_integral(self, start: float, end: float) -> float:
        st = self.state
        duration = st.blend_duration

        def primitive(offset: float) -> float:
            x = (st.blend_elapsed + offset) / duration
            return x - (x ** 3 - 0.5 * x ** 4)

        return duration * (primitive(end) - primitive(start))

    def _integrate_blend(self, clip, t0: float, start: float, span: float):
        st = self.state
        sums = [0.0, 0.0, 0.0]
        for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
            offset = start + node * span
            pose = clip.sample(t0 + offset)
            alpha = (st.blend_elapsed + offset) / st.blend_duration
            blend_weight = smoothstep(alpha)
            values = (
                pose.root_velocity[0],
                pose.root_velocity[1],
                pose.omega,
            )
            for i, value in enumerate(values):
                sums[i] += weight * blend_weight * value * span

        frozen_scale = self._frozen_weight_integral(start, end=start + span)
        return (
            st.frozen_velocity[0] * frozen_scale + sums[0],
            st.frozen_velocity[1] * frozen_scale + sums[1],
            st.frozen_omega * frozen_scale + sums[2],
        )

    def update(self, dt: float):
        if dt == 0.0:
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
            blended_dt = min(effective_dt, st.blend_duration - st.blend_elapsed)
            tail_dt = effective_dt - blended_dt
            breaks = self._velocity_breaks(clip, st.local_time, blended_dt)
            for start, end in zip(breaks, breaks[1:]):
                move = self._integrate_blend(
                    clip, st.local_time, start, end - start
                )
                dx += move[0]
                dy += move[1]
                dh += move[2]

            blend_end_time = st.local_time + blended_dt
            self._collect_events(clip, st.local_time, blend_end_time)
            st.local_time = blend_end_time
            st.blend_elapsed += blended_dt
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
            if st.blend_elapsed >= st.blend_duration:
                st.blend_elapsed = 0.0
                st.blend_duration = 0.0
                st.frozen_joints = None
                st.frozen_velocity = None
                st.frozen_omega = 0.0
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
            "transition_log": list(self._transition_log),
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
        self._transition_log = list(data.get("transition_log", []))

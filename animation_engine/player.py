import math
from dataclasses import dataclass

from .math_utils import clamp, lerp, smoothstep
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
    episode_start: float = 0.0
    frozen_joints: tuple = None
    frozen_velocity: tuple = None
    frozen_omega: float = 0.0
    queued_events: tuple = ()

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
                      entry_time: float = 0.0, sync_markers=()):
        """Start (or retarget) a blend.

        sync_markers maps a marker name on the current blended clip to one on
        the destination clip; the entry cursor is shifted so the destination
        phase matches the current phase (walk->run grounded foot). An
        explicit entry_time is honoured when no matching sync is given.

        Event progress rules while retargeting mid-blend:
        * events already drained from previous clips never replay;
        * queued-but-undrained events are kept and still delivered;
        * the destination event cursor starts at the chosen entry time, so
          events strictly ahead of it are not fabricated away, while events
          at or behind it are treated as already passed.
        """
        st = self.state
        if clip_name == st.clip_name and not st.is_blending:
            return
        snapshot = self.current_pose()
        target = self.library[clip_name]
        entry = entry_time
        if not target.loop:
            entry = min(entry, target.duration)
        for source_name, dest_name in sync_markers:
            try:
                source_time = self.clip.latest_marker_time(
                    source_name, st.local_time
                )
                offset = st.local_time - source_time
                dest_phase = target.marker(dest_name).t
            except KeyError:
                continue
            if target.loop:
                wanted = st.local_time - offset
                k = round((wanted - dest_phase) / target.duration)
                entry = dest_phase + k * target.duration + offset
            else:
                entry = min(dest_phase + offset, target.duration)
            break
        st.clip_name = clip_name
        st.local_time = entry
        st.episode_start = entry
        st.blend_elapsed = 0.0
        st.blend_duration = max(0.0, duration)
        st.frozen_joints = snapshot.joints
        st.frozen_velocity = snapshot.root_velocity
        st.frozen_omega = snapshot.omega

    def _cursor_breaks(self, clip, lo: float, hi: float):
        """Keyframe times and loop wraps strictly inside the unwrapped span."""
        points = set()
        if clip.loop:
            boundary = (math.floor(lo / clip.duration) + 1) * clip.duration
            while boundary < hi - 1e-12:
                if boundary > lo + 1e-12:
                    points.add(boundary)
                boundary += clip.duration
        for frame in clip.keyframes:
            phase = frame.t
            if clip.loop:
                k_first = math.floor((lo - phase) / clip.duration) + 1
                k_last = math.floor((hi - phase) / clip.duration)
                for k in range(k_first, k_last + 1):
                    candidate = phase + k * clip.duration
                    if lo + 1e-12 < candidate < hi - 1e-12:
                        points.add(candidate)
            elif lo + 1e-12 < phase < hi - 1e-12:
                points.add(phase)
        return points

    def _target_weight_primitive(self, cursor: float) -> float:
        st = self.state
        x = (cursor - st.episode_start) / st.blend_duration
        return x - (x ** 3 - 0.5 * x ** 4)

    def _integrate_blend(self, clip, a: float, b: float):
        """Exact frozen part + Gauss quadrature of the target root channels.

        Blend weight depends on the unwrapped target cursor: the blend began
        at episode_start and both cursor and blend elapsed advance together.
        Works for any span inside the active blend, forward or backward.
        """
        st = self.state
        span = b - a
        sums = [0.0, 0.0, 0.0]
        for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
            cursor = a + node * span
            pose = clip.sample(cursor)
            blend_weight = smoothstep(
                (cursor - st.episode_start) / st.blend_duration
            )
            for i, value in enumerate(
                (pose.root_velocity[0], pose.root_velocity[1], pose.omega)
            ):
                sums[i] += weight * blend_weight * value * span
        frozen_scale = st.blend_duration * (
            self._target_weight_primitive(b) - self._target_weight_primitive(a)
        )
        frozen_share = span - frozen_scale
        return (
            st.frozen_velocity[0] * frozen_share + sums[0],
            st.frozen_velocity[1] * frozen_share + sums[1],
            st.frozen_omega * frozen_share + sums[2],
        )

    def _segment_motion(self, clip, a: float, b: float, blended: bool):
        if blended:
            return self._integrate_blend(clip, a, b)
        return clip.integrate_root(a, b - a)

    def _apply_step(self, signed_dt: float):
        """Advance the cursor by signed_dt (already scaled by speed).

        Returns the event markers crossed while traversing forward.
        """
        st = self.state
        clip = self.clip
        old_cursor = st.local_time

        if signed_dt >= 0.0:
            upper = old_cursor + signed_dt
            if not clip.loop:
                upper = min(upper, clip.duration)
            lower = old_cursor
            forward = True
        else:
            lower = old_cursor + signed_dt
            if st.is_blending or st.blend_duration > 0.0:
                lower = max(lower, st.episode_start)
            elif not clip.loop:
                lower = max(lower, 0.0)
            upper = old_cursor
            forward = False

        if upper <= lower + 1e-15:
            return ()

        # Blending only covers the cursor range [episode_start, blend end].
        blend_end = st.episode_start + st.blend_duration
        boundaries = {lower, upper}
        if st.blend_duration > 0.0:
            boundaries.add(min(upper, blend_end))
        boundaries.update(self._cursor_breaks(clip, lower, upper))
        ordered = sorted(boundaries)

        # Forward segment motions and headings measured from the lower end,
        # so a backward traversal negates each segment exactly.
        motions = []
        headings_at = {lower: st.root_heading}
        heading = st.root_heading
        for a, b in zip(ordered, ordered[1:]):
            blended_span = st.blend_duration > 0.0 and b <= blend_end + 1e-12
            move = self._segment_motion(clip, a, b, blended_span)
            motions.append(move)
            heading += move[2]
            headings_at[b] = heading

        dx = dy = dh = 0.0
        if forward:
            for (a, b), move in zip(zip(ordered, ordered[1:]), motions):
                wx, wy = self._local_to_world_displacement(
                    headings_at[a], move[0], move[1], move[2]
                )
                dx += wx
                dy += wy
                dh += move[2]
            new_cursor = upper
        else:
            for (a, b), move in zip(zip(ordered, ordered[1:]), motions):
                wx, wy = self._local_to_world_displacement(
                    headings_at[a], move[0], move[1], move[2]
                )
                dx -= wx
                dy -= wy
                dh -= move[2]
            new_cursor = lower

        st.root_position = (
            st.root_position[0] + dx,
            st.root_position[1] + dy,
        )
        st.root_heading += dh

        if st.blend_duration > 0.0:
            st.blend_elapsed = new_cursor - st.episode_start
            if st.blend_elapsed >= st.blend_duration - 1e-12:
                self._finish_blend()
        st.local_time = new_cursor

        if forward:
            return clip.events_between(old_cursor, new_cursor)
        return ()

    def _finish_blend(self):
        st = self.state
        st.blend_elapsed = 0.0
        st.blend_duration = 0.0
        st.episode_start = 0.0
        st.frozen_joints = None
        st.frozen_velocity = None
        st.frozen_omega = 0.0

    def _enqueue(self, crossings, clip_name):
        st = self.state
        for _, marker in crossings:
            st.queued_events += ((clip_name, marker.name, marker.event),)

    def update(self, dt: float):
        if dt == 0.0:
            return self.current_pose()
        # Reverse playback is an explicit, supported direction: negative dt
        # (or negative speed) scrubs the timeline backwards, never fires
        # events, and re-arms crossed event markers for later forward passes.
        crossings = self._apply_step(dt * self.state.speed)
        self._enqueue(crossings, self.state.clip_name)
        return self.current_pose()

    def drain_events(self):
        """Return and clear this character's queued external events."""
        events = self.state.queued_events
        self.state.queued_events = ()
        return events

    def seek(self, cursor: float, *, fire_events: bool = False):
        """Jump the play head to an explicit unwrapped cursor.

        Any active blend is cancelled and the frozen snapshot is discarded.
        Forward jumps fire the crossed events in order (attack hit markers
        are not silently lost); pass fire_events=False to scrub past them.
        Backward jumps never fire and re-arm crossed markers. Root position
        and heading are untouched: a jump changes phase, not the character's
        place in the world.
        """
        st = self.state
        clip = self.clip
        target = cursor
        if not clip.loop:
            target = clamp(target, 0.0, clip.duration)
        elif target < 0.0:
            target = target % clip.duration
        crossings = ()
        if fire_events and target > st.local_time:
            crossings = clip.events_between(st.local_time, target)
        self._finish_blend()
        st.local_time = target
        self._enqueue(crossings, st.clip_name)
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
            "episode_start": st.episode_start,
            "frozen_joints": st.frozen_joints,
            "frozen_velocity": st.frozen_velocity,
            "frozen_omega": st.frozen_omega,
            "queued_events": st.queued_events,
        }

    def load_state(self, data: dict):
        blend_elapsed = data["blend_elapsed"]
        blend_duration = data["blend_duration"]
        default_episode = data["local_time"] - blend_elapsed if blend_duration else 0.0
        self.state = PlaybackState(
            clip_name=data["clip_name"],
            local_time=data["local_time"],
            speed=data["speed"],
            root_position=tuple(data["root_position"]),
            root_heading=data.get("root_heading", 0.0),
            blend_elapsed=data["blend_elapsed"],
            blend_duration=data["blend_duration"],
            episode_start=data.get("episode_start", default_episode),
            frozen_joints=data["frozen_joints"],
            frozen_velocity=data["frozen_velocity"],
            frozen_omega=data["frozen_omega"],
            queued_events=tuple(data.get("queued_events", ())),
        )

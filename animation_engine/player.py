import math
from dataclasses import dataclass

from .math_utils import lerp, smoothstep
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

    def transition_to(self, clip_name: str, duration: float, entry_time: float = 0.0):
        if clip_name == self.state.clip_name and not self.state.is_blending:
            return
        snapshot = self.current_pose()
        self.state.clip_name = clip_name
        self.state.local_time = entry_time
        self.state.blend_elapsed = 0.0
        self.state.blend_duration = max(0.0, duration)
        self.state.frozen_joints = snapshot.joints
        self.state.frozen_velocity = snapshot.root_velocity
        self.state.frozen_omega = snapshot.omega

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
        if dt < 0.0:
            raise ValueError("playback cannot move backwards")
        if dt == 0.0:
            return self.current_pose()

        st = self.state
        effective_dt = dt * st.speed
        clip = self.clip
        dx = dy = dh = 0.0

        if st.is_blending:
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

            st.local_time += blended_dt
            st.blend_elapsed += blended_dt
            if tail_dt > 0.0:
                move = clip.integrate_root(st.local_time, tail_dt)
                dx += move[0]
                dy += move[1]
                dh += move[2]
                st.local_time += tail_dt
            if st.blend_elapsed >= st.blend_duration:
                st.blend_elapsed = 0.0
                st.blend_duration = 0.0
                st.frozen_joints = None
                st.frozen_velocity = None
                st.frozen_omega = 0.0
        else:
            dx, dy, dh = clip.integrate_root(st.local_time, effective_dt)
            st.local_time += effective_dt

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
        )

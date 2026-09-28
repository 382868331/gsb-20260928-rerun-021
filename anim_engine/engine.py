"""Per-character animation playback over a shared clip library.

Transitions are weight-based cross-fades. Starting a new transition while a
previous one is unfinished only retargets existing layer weights, so the new
clip grows out of the current blended result with no pose or position pop.
"""

from .mathutil import quat_identity, quat_slerp, vec_scale


class _Layer:
    __slots__ = ("clip", "time", "weight", "target", "fade_rate", "speed")

    def __init__(self, clip, time, weight, target, fade_duration, speed):
        self.clip = clip
        self.time = time
        self.weight = weight
        self.target = target
        self.fade_rate = (1.0 / fade_duration) if fade_duration > 0.0 else 0.0
        self.speed = speed

    def to_state(self):
        return {
            "clip": self.clip.name,
            "time": self.time,
            "weight": self.weight,
            "target": self.target,
            "fade_rate": self.fade_rate,
            "speed": self.speed,
        }

    @classmethod
    def from_state(cls, data, library):
        layer = cls.__new__(cls)
        layer.clip = library[data["clip"]]
        layer.time = data["time"]
        layer.weight = data["weight"]
        layer.target = data["target"]
        layer.fade_rate = data["fade_rate"]
        layer.speed = data["speed"]
        return layer


class AnimInstance:
    """Independent playback state; clips are shared assets."""

    def __init__(self, library, clip=None, speed=1.0, position=(0.0, 0.0, 0.0),
                 event_callback=None):
        self.library = dict(library)
        self.elapsed = 0.0
        self.position = tuple(position)
        self.layers = []
        self.event_callback = event_callback
        if clip is not None:
            self.play(clip, speed=speed)

    def _resolve(self, clip):
        return clip if hasattr(clip, "name") else self.library[clip]

    def play(self, clip, speed=1.0):
        """Instantly start a clip with no blend."""
        clip = self._resolve(clip)
        self.layers = [_Layer(clip, 0.0, 1.0, 1.0, 0.0, speed)]

    def transition_to(self, clip, duration=0.3, speed=1.0):
        """Cross-fade to a clip; safe to re-call mid-fade.

        Every layer keeps its current weight and is retargeted to zero, while
        the incoming layer starts at zero: at the switch instant the blended
        pose and integrated root motion are provably unchanged.
        """
        clip = self._resolve(clip)
        rate = (1.0 / duration) if duration > 0.0 else 0.0
        for layer in self.layers:
            layer.target = 0.0
            layer.fade_rate = rate
        self.layers.append(_Layer(clip, 0.0, 0.0, 1.0, duration, speed))

    def set_speed(self, speed):
        for layer in self.layers:
            if layer.target == 1.0:
                layer.speed = speed

    def update(self, dt):
        """Advance dt seconds: times, fade weights and root motion together."""
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        self.elapsed += dt
        dx = 0.0
        for layer in self.layers:
            before = layer.time
            after = before + dt * layer.speed
            # Root movement is integrated from each clip's own cycle-aware
            # motion curve, then weighted -- never read off blended poses.
            dx += layer.weight * layer.clip.root_delta(before, after)
            layer.time = after
            if layer.fade_rate > 0.0:
                step = layer.fade_rate * dt
                if layer.target > layer.weight:
                    layer.weight = min(layer.target, layer.weight + step)
                else:
                    layer.weight = max(layer.target, layer.weight - step)
        self.position = (self.position[0] + dx, self.position[1],
                         self.position[2])
        if self.event_callback is not None:
            self._emit_events(dt)
        dead = [l for l in self.layers if l.weight <= 1e-7 and l.target == 0.0]
        if dead:
            keep = [l for l in self.layers if l not in dead]
            total = sum(l.weight for l in keep)
            if total > 0.0:
                for layer in keep:
                    layer.weight /= total
            self.layers = keep
        return self.sample_pose()

    def _emit_events(self, dt):
        # Extension point for footstep / expression events: a clip may carry
        # an events list of (local_time, name); crossing the playback head
        # fires it, including wrap crossings on looped clips.
        for layer in self.layers:
            events = getattr(layer.clip, "events", None)
            if not events:
                continue
            start = layer.time - dt * layer.speed
            end = layer.time
            duration = layer.clip.duration
            looped = layer.clip.wrap == "loop"
            for raw_t, name in events:
                fire_t = raw_t
                if looped and duration > 0.0:
                    fire_t = raw_t + int(start // duration) * duration
                    while fire_t <= start:
                        fire_t += duration
                if start < fire_t <= end:
                    self.event_callback(layer.clip.name, name, layer.weight)

    def sample_pose(self):
        """Weighted blend of all active layers (slerp rotations)."""
        if not self.layers:
            return {}
        total = sum(layer.weight for layer in self.layers)
        if total <= 1e-12:
            layer = self.layers[0]
            return layer.clip.sample_pose(layer.time)
        joints = set()
        for layer in self.layers:
            joints.update(layer.clip.tracks)
        sampled = [(layer.clip.sample_pose(layer.time), layer.weight / total)
                   for layer in self.layers]
        pose = {}
        for joint in sorted(joints):
            kind = next(iter({layer.clip.tracks[joint].kind
                              for layer in self.layers
                              if joint in layer.clip.tracks}))
            accum = None
            accum_w = 0.0
            for sample, weight in sampled:
                value = sample.get(joint)
                if value is None:
                    value = quat_identity() if kind == "rotation" else (0.0, 0.0, 0.0)
                if accum is None:
                    accum = value if kind == "rotation" else vec_scale(value, weight)
                elif kind == "rotation":
                    accum = quat_slerp(accum, value, weight / (accum_w + weight))
                else:
                    accum = (accum[0] + value[0] * weight,
                             accum[1] + value[1] * weight,
                             accum[2] + value[2] * weight)
                accum_w += weight
            pose[joint] = accum
        return pose

    def get_state(self):
        """JSON-serializable snapshot; clips referenced by name."""
        return {
            "elapsed": self.elapsed,
            "position": list(self.position),
            "layers": [layer.to_state() for layer in self.layers],
        }

    def set_state(self, state):
        """Restore a snapshot from get_state and keep sampling from it."""
        self.elapsed = state["elapsed"]
        self.position = tuple(state["position"])
        self.layers = [_Layer.from_state(d, self.library)
                       for d in state["layers"]]
        return self

"""Animation clip resources.

A clip is a *shared, stateless* asset: many character instances can play the
same clip with independent playback heads. It only knows how to sample poses
and root motion at a given local time; it never stores playback progress.
"""

from .mathutil import quat_slerp, vec_lerp


ROTATION = "rotation"
POSITION = "position"

LOOP = "loop"
CLAMP = "clamp"


class Track:
    """One animated channel for a joint (rotation quaternion or position)."""

    def __init__(self, joint, kind, keys):
        self.joint = joint
        self.kind = kind
        self.keys = sorted(keys, key=lambda kv: kv[0])

    def _mix(self, a, b, f):
        if self.kind == ROTATION:
            return quat_slerp(a, b, f)
        return vec_lerp(a, b, f)

    def sample(self, t, duration, wrap):
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            if wrap == LOOP and len(keys) > 1:
                frac = (t - keys[-1][0]) / duration if duration > 0.0 else 0.0
                return self._mix(keys[-1][1], keys[0][1], frac)
            return keys[-1][1]
        lo, hi = 0, len(keys) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if keys[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        (ta, va), (tb, vb) = keys[lo], keys[hi]
        frac = (t - ta) / (tb - ta) if tb > ta else 0.0
        return self._mix(va, vb, frac)


class AnimationClip:
    """Shared asset: per-joint tracks plus a cumulative root-motion track."""

    def __init__(self, name, duration, tracks, root_xs, wrap=LOOP):
        self.name = name
        self.duration = float(duration)
        self.tracks = {t.joint: t for t in tracks}
        self.root_xs = sorted(root_xs, key=lambda kv: kv[0])
        self.wrap = wrap

    def sample_pose(self, local_time):
        return {j: tr.sample(local_time, self.duration, self.wrap)
                for j, tr in self.tracks.items()}

    def root_x(self, local_time):
        xs = self.root_xs
        t = local_time
        if t <= xs[0][0]:
            return xs[0][1]
        if t >= xs[-1][0] and self.wrap != LOOP:
            return xs[-1][1]
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        (ta, va), (tb, vb) = xs[lo], xs[hi]
        frac = (t - ta) / (tb - ta) if tb > ta else 0.0
        return va + frac * (vb - va)

    def root_delta(self, t_before, t_after):
        """Forward displacement between two unwrapped local times.

        Full loop cycles each contribute one stride length, so root motion is
        continuous across the seam even under non-1.0 playback speeds.
        """
        if self.wrap != LOOP:
            return self.root_x(min(t_after, self.duration)) - self.root_x(t_before)
        stride = self.root_xs[-1][1] - self.root_xs[0][1]

        def parts(t):
            c = int(t // self.duration)
            return c, t - c * self.duration

        cb, rb = parts(t_before)
        ca, ra = parts(t_after)
        return (ca - cb) * stride + self.root_x(ra) - self.root_x(rb)


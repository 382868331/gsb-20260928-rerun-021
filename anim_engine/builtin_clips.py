"""Short built-in clips for demos and tests.

Three tiny procedural looped clips share one simple skeleton:
  idle - in place, gentle breathing            (loop, no root motion)
  walk - leg swing + bob, 0.6 m stride         (loop)
  run  - bigger swing, faster cadence, 1.2 m   (loop)

Each clip is deliberately short (~1 s) so a demo can switch again while the
first transition is still fading.
"""

import math

from .clip import AnimationClip, Track, ROTATION, LOOP
from .mathutil import quat_from_axis_angle


def _mul(a, b):
    return (
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


def _rot(x=0.0, y=0.0, z=0.0):
    q = (0.0, 0.0, 0.0, 1.0)
    for axis, angle in (((1, 0, 0), x), ((0, 1, 0), y), ((0, 0, 1), z)):
        q = _mul(q, quat_from_axis_angle(axis, angle))
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def _build_clip(name, duration, amp, stride, bob, samples=12):
    keys = {"hip": [], "l_leg": [], "r_leg": [], "l_arm": [], "r_arm": []}
    roots = []
    for i in range(samples + 1):
        t = duration * i / samples
        phase = 2.0 * math.pi * t / duration
        s = math.sin(phase)
        keys["hip"].append((t, _rot(z=bob * math.sin(2.0 * phase) * 0.5)))
        keys["l_leg"].append((t, _rot(x=amp * s)))
        keys["r_leg"].append((t, _rot(x=-amp * s)))
        keys["l_arm"].append((t, _rot(x=-amp * 0.8 * s)))
        keys["r_arm"].append((t, _rot(x=amp * 0.8 * s)))
        roots.append((t, stride * i / samples))
    tracks = [Track(j, ROTATION, k) for j, k in keys.items()]
    clip = AnimationClip(name, duration, tracks, roots, wrap=LOOP)
    # Extension data for future footstep / expression event dispatch.
    clip.events = [(duration * 0.25, "footstep_l"),
                   (duration * 0.75, "footstep_r")]
    return clip


def build_library():
    """Return the shared name -> AnimationClip library."""
    idle = _build_clip("idle", duration=1.0, amp=0.04, stride=0.0, bob=0.02)
    walk = _build_clip("walk", duration=0.8, amp=0.45, stride=0.6, bob=0.05)
    run = _build_clip("run", duration=0.5, amp=0.85, stride=1.2, bob=0.10)
    return {c.name: c for c in (idle, walk, run)}

"""Quaternion and vector helpers (pure Python, no dependencies).

Quaternions are (x, y, z, w) tuples. Vectors are (x, y, z) tuples.
"""

import math


def quat_identity():
    return (0.0, 0.0, 0.0, 1.0)


def quat_normalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return quat_identity()
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_slerp(a, b, t):
    """Shortest-path spherical interpolation."""
    d = quat_dot(a, b)
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    if d > 0.9995:
        return quat_normalize(tuple(a[i] + t * (b[i] - a[i]) for i in range(4)))
    d = max(-1.0, min(1.0, d))
    theta = math.acos(d)
    sin_t = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_t
    wb = math.sin(t * theta) / sin_t
    return tuple(wa * a[i] + wb * b[i] for i in range(4))


def quat_from_axis_angle(axis, angle):
    half = angle * 0.5
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def vec_length(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

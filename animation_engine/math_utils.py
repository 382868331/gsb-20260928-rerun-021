def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    """Cubic ease with zero first derivative at both endpoints."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend_vectors(a, b, weight: float):
    return tuple(lerp(x, y, weight) for x, y in zip(a, b))

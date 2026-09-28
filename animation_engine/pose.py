from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    """A sampled body state.

    joints are generic channel values (sufficient for the built-in demo and
    for tests); real assets can map the same channels onto a skeleton.
    root_velocity is expressed in the *clip local* frame, metres/second.
    omega is root heading change, radians/second.
    """

    joints: tuple
    root_velocity: tuple
    omega: float = 0.0

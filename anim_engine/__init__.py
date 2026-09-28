"""Character action orchestration engine.

Resources (AnimationClip) are shared assets; AnimInstance objects hold
independent playback state with condition-driven smooth transitions,
speed/loop/root-motion integration and savable state.
"""

from .clip import AnimationClip, Track, ROTATION, POSITION, LOOP, CLAMP
from .engine import AnimInstance
from .builtin_clips import build_library

__all__ = [
    "AnimationClip", "Track", "ROTATION", "POSITION", "LOOP", "CLAMP",
    "AnimInstance", "build_library",
]

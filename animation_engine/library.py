from .clip import AnimationClip, Keyframe, Marker


def build_demo_library():
    """Small procedural assets with deliberately varied root motion.

    Channels stand for two leg/body channels. The important part is that root
    velocity differs between actions, exposing a naive pose average as wrong.
    """
    clips = {
        "idle": AnimationClip(
            "idle",
            [
                Keyframe(0.0, (0.0, 0.0), (0.0, 0.0)),
                Keyframe(0.5, (0.04, -0.04), (0.0, 0.0)),
                Keyframe(1.0, (0.0, 0.0), (0.0, 0.0)),
            ],
            loop=True,
        ),
        "walk": AnimationClip(
            "walk",
            [
                Keyframe(0.0, (0.5, -0.5), (1.0, 0.0)),
                Keyframe(0.5, (-0.5, 0.5), (1.4, 0.0)),
                Keyframe(1.0, (0.5, -0.5), (1.0, 0.0)),
            ],
            loop=True,
            # Foot-down instants: channel zero hits its extreme.
            markers=[Marker("foot_down", 0.0), Marker("foot_down", 0.5)],
        ),
        "run": AnimationClip(
            "run",
            [
                Keyframe(0.0, (0.8, -0.8), (3.0, 0.0)),
                Keyframe(0.5, (-0.8, 0.8), (3.6, 0.0)),
                Keyframe(1.0, (0.8, -0.8), (3.0, 0.0)),
            ],
            loop=True,
            markers=[Marker("foot_down", 0.0), Marker("foot_down", 0.5)],
        ),
        # Short one-shot action, intentionally started while a blend is active.
            "step_dodge": AnimationClip(
            "step_dodge",
            [
                Keyframe(0.0, (0.0, 0.0), (0.2, 0.4)),
                Keyframe(0.15, (-0.7, 0.7), (0.4, 1.2)),
                Keyframe(0.30, (0.7, -0.7), (0.0, 0.0)),
            ],
            loop=False,
        ),
        # One-shot attack: only the marker moment tells the world to hit.
        "attack": AnimationClip(
            "attack",
            [
                Keyframe(0.0, (0.0, 0.0), (0.0, 0.0)),
                Keyframe(0.12, (0.3, 0.0), (0.2, 0.0)),
                Keyframe(0.2, (0.9, 0.0), (0.4, 0.0)),
                Keyframe(0.34, (0.1, 0.0), (0.0, 0.0)),
            ],
            loop=False,
            markers=[Marker("strike", 0.2, event={"type": "hit", "damage": 10})],
        ),
    }
    return clips

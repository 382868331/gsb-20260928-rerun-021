import math
import unittest

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    State,
    build_demo_library,
)


class EngineTests(unittest.TestCase):
    def test_shared_clips_have_independent_players(self):
        library = build_demo_library()
        a = CharacterAnimator(library, "walk", [State("walk")])
        b = CharacterAnimator(library, "walk", [State("walk")])
        a.update(0.3)
        self.assertAlmostEqual(a.player.state.local_time, 0.3)
        self.assertEqual(b.player.state.local_time, 0.0)

    def test_loop_root_motion_integrates_velocity(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.update(2.5)
        x, _ = animator.root_position
        # Walk velocity is 1.0 then 1.4 over each one-second loop.
        self.assertAlmostEqual(x, 2.5 * 1.2)

    def test_retransition_has_no_pose_or_root_jump(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "idle",
            [
                State("idle"),
                State("walk"),
                State("dodge", "step_dodge"),
            ],
        )
        before_pose = animator.current_pose()
        before_root = animator.root_position
        animator.request_transition("walk", 0.4)
        animator.update(0.2)
        interrupted_pose = animator.current_pose()
        interrupted_root = animator.root_position
        animator.request_transition("dodge", 0.2)
        after_pose = animator.current_pose()
        after_root = animator.root_position

        for old, new in zip(interrupted_pose.joints, after_pose.joints):
            self.assertAlmostEqual(old, new)
        for old, new in zip(interrupted_pose.root_velocity, after_pose.root_velocity):
            self.assertAlmostEqual(old, new)
        self.assertGreater(
            (interrupted_root[0] - before_root[0]) ** 2
            + (interrupted_root[1] - before_root[1]) ** 2,
            0.0,
        )
        self.assertEqual(after_root, interrupted_root)
        self.assertTrue(math.isfinite(before_pose.joints[0]))

    def test_save_resume_matches_uninterrupted_sampling(self):
        library = build_demo_library()
        states = [State("idle"), State("walk"), State("run")]
        continuous = CharacterAnimator(library, "idle", states)
        restored = CharacterAnimator(library, "idle", states)

        continuous.request_transition("walk", 0.4)
        restored.request_transition("walk", 0.4)
        for _ in range(3):
            continuous.update(0.1)
            restored.update(0.1)
        continuous.request_transition("run", 0.5)
        restored.request_transition("run", 0.5)
        continuous.update(0.1)
        restored.update(0.1)

        snapshot = restored.save_state()
        for _ in range(8):
            continuous.update(0.07)
        restored.load_state(snapshot)
        for _ in range(8):
            restored.update(0.07)

        self.assertAlmostEqual(
            continuous.player.state.local_time,
            restored.player.state.local_time,
            places=10,
        )
        for a, b in zip(continuous.root_position, restored.root_position):
            self.assertAlmostEqual(a, b, places=10)
        for a, b in zip(continuous.current_pose().joints,
                        restored.current_pose().joints):
            self.assertAlmostEqual(a, b, places=10)

    def test_conditional_transition(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "idle",
            [State("idle"), State("walk")],
            [],
        )
        from animation_engine import Transition

        animator.add_transition(
            Transition("idle", "walk", 0.2,
                       condition=lambda ctx: ctx["moving"])
        )
        animator.update(0.1, {"moving": False})
        self.assertEqual(animator.current_state, "idle")
        animator.update(0.1, {"moving": True})
        self.assertEqual(animator.current_state, "walk")

    def test_root_heading_is_saved_and_applied_to_world_motion(self):
        library = {
            "turn_walk": AnimationClip(
                "turn_walk",
                [
                    Keyframe(0.0, (0.0,), (1.0, 0.0), omega=1.0),
                    Keyframe(1.0, (1.0,), (1.0, 0.0), omega=1.0),
                ],
                loop=False,
            )
        }
        animator = CharacterAnimator(
            library, "turn_walk", [State("turn_walk")]
        )
        animator.update(1.0)
        self.assertAlmostEqual(animator.root_heading, 1.0)
        self.assertAlmostEqual(animator.root_position[0], math.sin(1.0))
        self.assertAlmostEqual(animator.root_position[1], 1.0 - math.cos(1.0))

        restored = CharacterAnimator(
            library, "turn_walk", [State("turn_walk")]
        )
        restored.load_state(animator.save_state())
        self.assertAlmostEqual(restored.root_heading, 1.0)


if __name__ == "__main__":
    unittest.main()

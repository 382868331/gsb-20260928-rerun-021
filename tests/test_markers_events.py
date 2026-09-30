import math
import unittest

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    Marker,
    State,
    build_demo_library,
)


def _attack_animator(library):
    return CharacterAnimator(
        library,
        "idle",
        [State("idle"), State("attack")],
    )


class MarkerSyncTests(unittest.TestCase):
    def test_loop_sync_picks_nearest_tagged_phase(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library, "walk", [State("walk"), State("run")]
        )
        animator.update(0.63)
        animator.request_transition("run", 0.3, sync_marker="foot_down")
        # Nearest run foot-down occurrence to walk time 0.63 is t=1.0.
        self.assertAlmostEqual(animator.player.state.local_time, 1.0)

    def test_sync_does_not_replay_entry_marker_at_time_zero(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library, "walk", [State("walk"), State("run")]
        )
        animator.request_transition("run", 0.3, sync_marker="foot_down")
        self.assertEqual(animator.pop_events(), [])


class EventProgressTests(unittest.TestCase):
    def test_big_step_and_frame_steps_emit_same_events(self):
        library = build_demo_library()

        big = _attack_animator(library)
        big.request_transition("attack", 0.0)
        big.update(0.6)
        big_events = big.pop_events()

        stepped = _attack_animator(library)
        stepped.request_transition("attack", 0.0)
        stepped_events = []
        for _ in range(12):
            stepped.update(0.05)
            stepped_events.extend(stepped.pop_events())

        self.assertEqual(
            [(e["marker"], e["payload"]) for e in big_events],
            [("windup", {"stage": "anticipation"}),
             ("hit", {"damage": 10})],
        )
        self.assertEqual(
            [e["marker"] for e in stepped_events],
            [e["marker"] for e in big_events],
        )

    def test_loop_marker_fires_once_per_cycle(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.update(2.5)
        events = animator.pop_events()
        self.assertEqual([e["cycle"] for e in events], [1, 2])
        self.assertEqual(len(events), 2)
        # Standing at / advancing within the same phase fires nothing again.
        animator.update(0.1)
        self.assertEqual(animator.pop_events(), [])

    def test_shared_clips_keep_independent_event_progress(self):
        library = build_demo_library()
        a = _attack_animator(library)
        b = _attack_animator(library)
        a.request_transition("attack", 0.0)
        b.request_transition("attack", 0.0)
        a.update(0.4)
        self.assertEqual([e["marker"] for e in a.pop_events()],
                         ["windup", "hit"])
        b.update(0.1)
        self.assertEqual(b.pop_events(), [])
        b.update(0.4)
        self.assertEqual([e["marker"] for e in b.pop_events()],
                         ["windup", "hit"])

    def test_retransition_keeps_queued_events_and_cancels_unreached(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "idle",
            [State("idle"), State("attack"), State("walk")],
        )
        animator.request_transition("attack", 0.3)
        animator.update(0.28)  # blend still active; windup already fired
        self.assertEqual(
            [e["marker"] for e in animator.player.save_state()["pending_events"]],
            ["windup"],
        )
        animator.request_transition("walk", 0.2)
        pending = animator.player.save_state()["pending_events"]
        # Queued external event is delivered even after switching away.
        self.assertEqual(len(pending), 1)
        animator.update(1.0)
        markers = [e["marker"] for e in animator.pop_events()]
        self.assertEqual(markers[0], "windup")
        # attack 'hit' is abandoned with the clip; walk foot markers remain.
        self.assertNotIn("hit", markers)
        self.assertIn("foot_down", markers)

    def test_already_fired_event_does_not_replay(self):
        library = build_demo_library()
        animator = _attack_animator(library)
        animator.request_transition("attack", 0.0)
        animator.update(0.4)
        animator.pop_events()
        animator.seek(0.0)
        animator.update(0.4)
        self.assertEqual(animator.pop_events(), [])


class ReverseAndSeekTests(unittest.TestCase):
    def test_reverse_rewinds_root_and_pose_without_events(self):
        clip = AnimationClip(
            "slide",
            [
                Keyframe(0.0, (0.0,), (1.0, 0.0)),
                Keyframe(1.0, (1.0,), (1.0, 0.0)),
            ],
            loop=False,
            markers=[Marker(0.5, "tick")],
        )
        animator = CharacterAnimator(
            {"slide": clip}, "slide", [State("slide")]
        )
        animator.update(0.7)
        self.assertEqual([e["marker"] for e in animator.pop_events()],
                         ["tick"])
        x_forward = animator.root_position[0]
        animator.update(-0.7)
        self.assertAlmostEqual(animator.player.state.local_time, 0.0)
        self.assertAlmostEqual(animator.root_position[0], 0.0)
        self.assertLess(animator.root_position[0], x_forward)
        self.assertEqual(animator.pop_events(), [])
        # Re-crossing a fired marker while resuming forward does not replay.
        animator.update(0.7)
        self.assertEqual(animator.pop_events(), [])

    def test_seek_emits_no_events_and_cancels_blend(self):
        library = build_demo_library()
        animator = _attack_animator(library)
        animator.request_transition("attack", 0.3)
        animator.seek(0.5)  # jump past windup and hit
        self.assertFalse(animator.player.state.is_blending)
        self.assertEqual(animator.pop_events(), [])
        animator.update(0.05)
        self.assertEqual(animator.pop_events(), [])

    def test_reverse_during_blend_cancels_blend(self):
        library = build_demo_library()
        animator = _attack_animator(library)
        animator.request_transition("attack", 0.4)
        animator.update(0.1)
        self.assertTrue(animator.player.state.is_blending)
        animator.update(-0.05)
        self.assertFalse(animator.player.state.is_blending)
        self.assertEqual(animator.pop_events(), [])


class SaveResumeEventTests(unittest.TestCase):
    def test_save_resume_matches_pose_root_and_event_sequence(self):
        library = build_demo_library()
        states = [State("idle"), State("walk"), State("attack")]
        continuous = CharacterAnimator(library, "idle", states)
        restored = CharacterAnimator(library, "idle", states)
        for animator in (continuous, restored):
            animator.request_transition("walk", 0.2)
        continuous.update(0.3)
        restored.update(0.3)
        for animator in (continuous, restored):
            animator.request_transition("attack", 0.4)
        continuous.update(0.1)
        restored.update(0.1)

        snapshot = restored.save_state()

        def run(animator):
            events = []
            animator.pop_events()
            # Big step crosses both markers inside a still-active blend.
            animator.update(0.45)
            events.extend(animator.pop_events())
            animator.update(0.1)
            events.extend(animator.pop_events())
            return events

        expected = run(continuous)
        restored.load_state(snapshot)
        actual = run(restored)

        self.assertEqual(
            [e["marker"] for e in expected],
            [e["marker"] for e in actual],
        )
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


if __name__ == "__main__":
    unittest.main()

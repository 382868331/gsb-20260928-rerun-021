import math
import unittest

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    Marker,
    State,
    Transition,
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


class MarkerSyncTests(unittest.TestCase):
    def test_walk_to_run_aligns_grounded_foot_phase(self):
        library = build_demo_library()
        walk = CharacterAnimator(library, "walk", [State("walk"), State("run")])
        # Phase 0.37 inside the walk stride; foot_down repeats every 0.5.
        walk.player.state.local_time = 1.37
        walk.request_transition(
            "run", 0.3,
            sync_markers=(("foot_down", "foot_down"),),
        )
        entry = walk.player.state.local_time
        self.assertAlmostEqual(entry % 0.5, 1.37 % 0.5, places=12)

    def test_sync_transition_via_state_machine(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "walk",
            [State("walk"), State("run")],
            [Transition("walk", "run", 0.3,
                        condition=lambda ctx: ctx.get("faster"),
                        sync_markers=(("foot_down", "foot_down"),))],
        )
        animator.player.state.local_time = 2.1
        animator.update(0.01, {"faster": True})
        self.assertEqual(animator.current_state, "run")
        self.assertAlmostEqual(
            animator.player.state.local_time % 0.5, 0.11, places=10
        )


class EventTests(unittest.TestCase):
    def test_event_fires_only_when_marker_is_crossed(self):
        library = build_demo_library()
        attacker = CharacterAnimator(
            library, "attack", [State("attack")]
        )
        attacker.update(0.19)
        self.assertEqual(attacker.drain_events(), ())
        attacker.update(0.02)
        events = attacker.drain_events()
        self.assertEqual(len(events), 1)
        clip_name, marker_name, payload = events[0]
        self.assertEqual((clip_name, marker_name), ("attack", "strike"))
        self.assertEqual(payload["type"], "hit")
        # After draining, further playback never replays the same event.
        attacker.update(0.1)
        self.assertEqual(attacker.drain_events(), ())

    def test_big_step_and_per_frame_steps_produce_same_events(self):
        library = build_demo_library()
        stepped = CharacterAnimator(library, "attack", [State("attack")])
        chunked = CharacterAnimator(library, "attack", [State("attack")])

        stepped.update(0.3)
        for _ in range(30):
            chunked.update(0.01)

        self.assertEqual(stepped.drain_events(), chunked.drain_events())
        self.assertAlmostEqual(
            stepped.player.state.local_time,
            chunked.player.state.local_time,
            places=12,
        )

    def test_looping_events_fire_every_cycle(self):
        library = {
            "pulse": AnimationClip(
                "pulse",
                [Keyframe(0.0, (0.0,)), Keyframe(1.0, (1.0,))],
                loop=True,
                markers=[Marker("tick", 0.5, event="pulse")],
            )
        }
        animator = CharacterAnimator(library, "pulse", [State("pulse")])
        animator.update(2.6)
        self.assertEqual(
            [name for _, name, _ in animator.drain_events()],
            ["tick", "tick", "tick"],
        )

    def test_retransition_keeps_undrained_events_and_does_not_replay(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "attack",
            [State("attack"), State("run"), State("attack")],
        )
        animator.update(0.25)
        fired = animator.drain_events()
        self.assertEqual(len(fired), 1)
        # Switch away and back: the already-fired hit must not replay.
        animator.request_transition("run", 0.1)
        animator.update(0.2)
        animator.request_transition("attack", 0.1, entry_time=0.0)
        animator.update(0.25)
        self.assertEqual(len(animator.drain_events()), 1)

    def test_events_queued_before_switch_are_still_delivered(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "attack",
            [State("attack"), State("run")],
        )
        animator.update(0.25)
        animator.request_transition("run", 0.1)
        events = animator.drain_events()
        self.assertEqual([name for _, name, _ in events], ["strike"])

    def test_entry_time_past_marker_does_not_fabricate_event(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "run",
            [State("run"), State("attack")],
        )
        animator.request_transition("attack", 0.1, entry_time=0.25)
        animator.update(0.1)
        self.assertEqual(animator.drain_events(), ())

    def test_events_before_entry_are_lost_but_future_ones_fire(self):
        # Entry at 0.1: strike at 0.2 is still ahead and must fire.
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "run",
            [State("run"), State("attack")],
        )
        animator.request_transition("attack", 0.0, entry_time=0.1)
        animator.update(0.15)
        self.assertEqual(
            [name for _, name, _ in animator.drain_events()], ["strike"]
        )

    def test_shared_clips_keep_independent_event_progress(self):
        library = build_demo_library()
        a = CharacterAnimator(library, "attack", [State("attack")])
        b = CharacterAnimator(library, "attack", [State("attack")])
        a.update(0.25)
        self.assertEqual(len(a.drain_events()), 1)
        self.assertEqual(b.drain_events(), ())
        b.update(0.25)
        self.assertEqual(len(b.drain_events()), 1)


class ReverseAndSeekTests(unittest.TestCase):
    def test_reverse_does_not_fire_and_rearms_marker(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library, "attack", [State("attack")]
        )
        animator.update(0.25)
        self.assertEqual(len(animator.drain_events()), 1)
        animator.player.state.speed = -1.0
        animator.update(0.2)
        self.assertLess(animator.player.state.local_time, 0.2)
        self.assertEqual(animator.drain_events(), ())
        animator.player.state.speed = 1.0
        animator.update(0.25)
        # Re-crossed forward: the marker legitimately fires again.
        self.assertEqual(len(animator.drain_events()), 1)

    def test_negative_dt_scrubs_backwards(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.update(0.6)
        pose_later = animator.current_pose()
        animator.update(-0.3)
        self.assertAlmostEqual(animator.player.state.local_time, 0.3, places=12)
        self.assertEqual(
            tuple(round(v, 12) for v in animator.current_pose().joints),
            tuple(round(v, 12) for v in
                  library["walk"].sample(0.3).joints),
        )

    def test_reverse_root_motion_returns_to_start(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.update(1.0)
        moved = animator.root_position
        self.assertNotEqual(moved, (0.0, 0.0))
        animator.update(-1.0)
        self.assertAlmostEqual(animator.player.state.local_time, 0.0, places=10)
        self.assertAlmostEqual(animator.root_position[0], 0.0, places=10)
        self.assertAlmostEqual(animator.root_position[1], 0.0, places=10)

    def test_seek_forward_fires_crossed_events_backward_seek_arms(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "attack", [State("attack")])
        animator.player.seek(0.25)
        self.assertEqual(animator.drain_events(), ())
        animator.player.seek(0.25, fire_events=True)
        self.assertEqual(animator.drain_events(), ())
        animator.player.seek(0.0)
        animator.player.seek(0.3, fire_events=True)
        self.assertEqual(
            [name for _, name, _ in animator.drain_events()], ["strike"]
        )
        animator.player.seek(0.0)
        self.assertEqual(animator.drain_events(), ())

    def test_seek_cancels_active_blend(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library,
            "idle",
            [State("idle"), State("walk")],
        )
        animator.request_transition("walk", 0.5)
        animator.update(0.1)
        self.assertTrue(animator.player.state.is_blending)
        animator.player.seek(0.4)
        self.assertFalse(animator.player.state.is_blending)
        self.assertAlmostEqual(animator.player.state.local_time, 0.4)


class SavedEventsTests(unittest.TestCase):
    def test_save_restore_preserves_event_progress(self):
        library = build_demo_library()
        states = [State("attack"), State("run")]
        animator = CharacterAnimator(library, "attack", states)
        animator.update(0.1)
        data = animator.save_state()
        restored = CharacterAnimator(library, "attack", states)
        restored.load_state(data)
        restored.update(0.2)
        self.assertEqual(
            [name for _, name, _ in restored.drain_events()], ["strike"]
        )
        restored.update(0.1)
        self.assertEqual(restored.drain_events(), ())

    def test_save_restore_keeps_queued_events(self):
        library = build_demo_library()
        states = [State("attack")]
        animator = CharacterAnimator(library, "attack", states)
        animator.update(0.25)
        restored = CharacterAnimator(library, "attack", states)
        restored.load_state(animator.save_state())
        self.assertEqual(
            [name for _, name, _ in restored.drain_events()], ["strike"]
        )


if __name__ == "__main__":
    unittest.main()

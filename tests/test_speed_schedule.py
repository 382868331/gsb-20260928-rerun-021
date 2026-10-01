import unittest

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    Marker,
    State,
    build_demo_library,
)


def _gait_clip(name, duration, marker_t=0.0):
    return AnimationClip(
        name,
        [
            Keyframe(0.0, (0.0,), (1.0, 0.0)),
            Keyframe(duration, (1.0,), (1.0, 0.0)),
        ],
        loop=True,
        markers=[Marker(marker_t, "foot_down", {"foot": "left"})],
    )


def _animator(clips, initial):
    return CharacterAnimator(
        clips, initial, [State(name) for name in clips]
    )


class PhaseProportionalSyncTests(unittest.TestCase):
    def test_shorter_target_cycle_lands_on_matching_phase(self):
        clips = {"long": _gait_clip("long", 1.2), "short": _gait_clip("short", 0.6)}
        animator = _animator(clips, "long")
        animator.update(0.72)
        animator.request_transition("short", 0.2, sync_marker="foot_down")
        # 0.72 / 1.2 = 0.6 of the source cycle -> 0.6 * 0.6 = 0.36.
        self.assertAlmostEqual(animator.player.state.local_time, 0.36)

    def test_nonzero_marker_and_loop_boundary_keep_phase(self):
        clips = {
            "long": _gait_clip("long", 1.2, marker_t=0.3),
            "short": _gait_clip("short", 0.6, marker_t=0.3),
        }
        long_clip, short_clip = clips["long"], clips["short"]
        # 60% through the post-marker cycle, one full loop later.
        expected = (0.3 + 0.72 * 0.5) % 0.6
        self.assertAlmostEqual(
            short_clip.sync_time("foot_down", long_clip, 0.3 + 0.72),
            expected,
        )
        self.assertAlmostEqual(
            short_clip.sync_time("foot_down", long_clip, 0.3 + 0.72 + 1.2),
            expected,
        )
        # Reverse-playback time uses the same modulo phase.
        self.assertAlmostEqual(
            short_clip.sync_time("foot_down", long_clip, 0.3 + 0.72 - 1.2),
            expected,
        )

    def test_equal_periods_still_carry_the_absolute_offset(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library, "walk", [State("walk"), State("run")]
        )
        animator.update(0.63)
        animator.request_transition("run", 0.3, sync_marker="foot_down")
        self.assertAlmostEqual(animator.player.state.local_time, 0.63)


class SpeedScheduleTests(unittest.TestCase):
    def test_segment_rates_shift_marker_arrival_times(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        # Stance half at 0.5x, swing half at 2.0x: cycle takes 1.0 + 0.25.
        animator.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        animator.update(1.0)
        self.assertEqual(animator.pop_events(), [])  # cycle not finished
        self.assertAlmostEqual(animator.player.state.local_time, 0.5)
        animator.update(0.25)
        events = animator.pop_events()
        self.assertEqual([e["marker"] for e in events], ["foot_down"])
        self.assertAlmostEqual(animator.player.state.local_time, 1.0)

    def test_target_duration_stretches_full_cycle(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.set_speed_schedule(
            [(0.5, 0.5), (1.0, 2.0)], target_duration=0.8
        )
        animator.update(0.8)
        events = animator.pop_events()
        self.assertEqual([e["marker"] for e in events], ["foot_down"])
        self.assertAlmostEqual(animator.player.state.local_time, 1.0)

    def test_distance_per_cycle_matches_uniform_playback(self):
        library = build_demo_library()
        uniform = CharacterAnimator(library, "walk", [State("walk")])
        warped = CharacterAnimator(library, "walk", [State("walk")])
        warped.set_speed_schedule(
            [(0.5, 0.5), (1.0, 2.0)], target_duration=0.8
        )
        for _ in range(3):
            uniform.update(1.0)
            warped.update(0.8)
        for axis in range(2):
            self.assertAlmostEqual(
                uniform.root_position[axis],
                warped.root_position[axis],
                places=9,
            )

    def test_schedule_change_mid_blend_keeps_fired_events(self):
        library = build_demo_library()
        animator = CharacterAnimator(
            library, "walk", [State("walk"), State("run")]
        )
        animator.update(0.63)
        animator.request_transition("run", 0.5, sync_marker="foot_down")
        animator.update(0.37)  # blend active; one run foot_down fired
        fired = [e["cycle"] for e in animator.pop_events()]
        self.assertEqual(fired, [1])
        # New arrangement only affects time not yet played.
        animator.set_speed_schedule([(0.5, 0.25), (1.0, 4.0)])
        animator.update(1.0)  # 0.25 clip s at 0.25x -> phase 0.25
        self.assertEqual(animator.pop_events(), [])
        animator.update(1.0)  # reaches phase 0.5, still no wrap
        self.assertEqual(animator.pop_events(), [])
        animator.update(0.125)  # 4.0x carries the cursor to the wrap
        events = animator.pop_events()
        self.assertEqual([e["marker"] for e in events], ["foot_down"])
        self.assertEqual([e["cycle"] for e in events], [2])

    def test_schedule_survives_save_restore_mid_blend(self):
        library = build_demo_library()
        states = [State("walk"), State("run")]
        continuous = CharacterAnimator(library, "walk", states)
        continuous.update(0.63)
        continuous.request_transition("run", 0.5, sync_marker="foot_down")
        continuous.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        continuous.update(0.2)
        checkpoint = continuous.save_state()

        restored = CharacterAnimator(library, "walk", states)
        restored.load_state(checkpoint)
        events_a, events_b = [], []
        for _ in range(20):
            continuous.update(0.05)
            restored.update(0.05)
            events_a.extend(continuous.pop_events())
            events_b.extend(restored.pop_events())
        self.assertEqual(
            [e["marker"] for e in events_a],
            [e["marker"] for e in events_b],
        )
        self.assertAlmostEqual(
            continuous.player.state.local_time,
            restored.player.state.local_time,
            places=10,
        )
        for a, b in zip(continuous.root_position, restored.root_position):
            self.assertAlmostEqual(a, b, places=10)

    def test_third_action_switch_continues_warped_phase(self):
        library = build_demo_library()
        states = [State("walk"), State("run"), State("stride", "walk")]
        animator = CharacterAnimator(library, "walk", states)
        animator.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        animator.update(0.6)  # 0.6 wall s at 0.5x -> phase 0.3
        self.assertAlmostEqual(animator.player.state.local_time, 0.3)
        animator.request_transition("run", 0.3, sync_marker="foot_down")
        # Same cycle length: the warped phase carries over unchanged.
        self.assertAlmostEqual(animator.player.state.local_time, 0.3)
        animator.update(0.2)
        animator.request_transition("stride", 0.2, sync_marker="foot_down")
        contact = animator.contact_state()
        self.assertAlmostEqual(contact["offset"], 0.5)
        self.assertFalse(contact["grounded"] is None)

    def test_reverse_playback_uses_schedule_without_events(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", [State("walk")])
        animator.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        animator.update(1.25)  # one full warped cycle
        self.assertEqual(len(animator.pop_events()), 1)
        animator.update(-1.25)
        self.assertAlmostEqual(animator.player.state.local_time, 0.0)
        self.assertEqual(animator.pop_events(), [])


if __name__ == "__main__":
    unittest.main()

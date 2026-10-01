import unittest

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    Marker,
    State,
)


def _gait_clip(name, duration, marker_t=0.0, speed=1.0, extra_markers=()):
    return AnimationClip(
        name,
        [
            Keyframe(0.0, (0.0,), (speed, 0.0)),
            Keyframe(duration, (0.0,), (speed, 0.0)),
        ],
        loop=True,
        markers=[Marker(marker_t, "foot_down")] + list(extra_markers),
    )


def _animator(library, initial, extra_states=()):
    return CharacterAnimator(
        library,
        initial,
        [State(name) for name in library] + list(extra_states),
    )


class DifferentPeriodSyncTests(unittest.TestCase):
    def setUp(self):
        self.slow = _gait_clip("slow", 1.2)
        self.fast = _gait_clip("fast", 0.6)
        self.library = {"slow": self.slow, "fast": self.fast}

    def test_switch_scales_phase_into_shorter_cycle(self):
        animator = _animator(self.library, "slow")
        animator.update(0.72)
        animator.request_transition("fast", 0.2, sync_marker="foot_down")
        # 0.72/1.2 = 60% of the slow cycle must land at 60% of the fast
        # cycle (0.36), not at the raw-offset wrap of 0.12.
        self.assertAlmostEqual(animator.player.state.local_time, 0.36)

    def test_nonzero_marker_and_loop_boundary_keep_phase(self):
        slow = _gait_clip("slow", 1.2, marker_t=0.3)
        fast = _gait_clip("fast", 0.6, marker_t=0.3)
        # 0.72 is 0.42 past the marker; one loop later (1.92) is the same
        # contact phase and must map identically.
        for now in (0.72, 1.92):
            synced = fast.sync_time("foot_down", slow, now)
            phase = (0.72 - 0.3) / 1.2
            self.assertAlmostEqual(synced, 0.3 + phase * 0.6)

    def test_reverse_playback_time_uses_same_phase(self):
        # -0.48 on the slow loop is the same phase as +0.72.
        synced = self.fast.sync_time("foot_down", self.slow, -0.48)
        self.assertAlmostEqual(synced, 0.36)

    def test_contact_snapshot_progress_matches_across_switch(self):
        animator = _animator(self.library, "slow")
        animator.update(0.72)
        animator.request_transition("fast", 0.2, sync_marker="foot_down")
        entry = animator.transition_history()[-1]
        self.assertAlmostEqual(
            entry["before"]["progress"], entry["after"]["progress"]
        )
        self.assertEqual(
            entry["before"]["grounded"], entry["after"]["grounded"]
        )


class SpeedScheduleTests(unittest.TestCase):
    def _library(self):
        clip = _gait_clip(
            "gait", 1.0, extra_markers=[Marker(0.5, "mid")]
        )
        return {"gait": clip}

    @staticmethod
    def _event_times(animator, wall_budget, dt=0.01):
        times = {}
        wall = 0.0
        while wall < wall_budget - 1e-12:
            animator.update(dt)
            wall += dt
            for event in animator.pop_events():
                times.setdefault(event["marker"], []).append(wall)
        return times

    def test_segments_shift_marker_arrival_and_hit_target_cycle(self):
        animator = _animator(self._library(), "gait")
        # Stance half slowed (0.5x), swing half sped up (2x), one full
        # cycle normalised to exactly 1.0 s of wall time.
        animator.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        # scale = duration * (0.5/0.5 + 0.5/2.0) / 1.0 = 1.25, so the first
        # half runs at 0.625 clip-seconds per wall second.
        # Event times are quantised to the 0.01 s update frame.
        times = self._event_times(animator, 1.05)
        self.assertAlmostEqual(times["mid"][0], 0.8, delta=0.015)
        self.assertAlmostEqual(times["foot_down"][0], 1.0, delta=0.015)

    def test_uniform_and_variable_cover_same_distance_per_cycle(self):
        uniform = _animator(self._library(), "gait")
        variable = _animator(self._library(), "gait")
        variable.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        uniform_times = self._event_times(uniform, 1.05)
        variable_times = self._event_times(variable, 1.05)
        # Same distance after exactly one 1.0 s cycle.
        uniform_cycle = _animator(self._library(), "gait")
        variable_cycle = _animator(self._library(), "gait")
        variable_cycle.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        for _ in range(100):
            uniform_cycle.update(0.01)
            variable_cycle.update(0.01)
        self.assertAlmostEqual(
            uniform_cycle.root_position[0],
            variable_cycle.root_position[0],
            places=7,
        )
        self.assertAlmostEqual(uniform_times["mid"][0], 0.5, delta=0.02)
        self.assertNotAlmostEqual(
            uniform_times["mid"][0], variable_times["mid"][0], places=3
        )
        # Both complete the cycle at the 1.0 s target (frame quantised).
        for times in (uniform_times, variable_times):
            self.assertAlmostEqual(
                times["foot_down"][0], 1.0, delta=0.02
            )

    def test_schedule_change_mid_blend_retimes_future_events_only(self):
        library = self._library()
        library["other"] = _gait_clip("other", 1.0)
        animator = _animator(library, "other")
        animator.request_transition("gait", 0.5)
        animator.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        animator.update(0.4)  # clip time 0.25 under the slowed first half
        self.assertEqual(animator.pop_events(), [])
        # Speed the whole cycle up mid-blend: remaining 0.25 of phase now
        # runs at 2.0 clip-seconds per wall second.
        animator.set_speed_schedule([(0.0, 2.0)], target_cycle_duration=0.5)
        times = self._event_times(animator, 1.0)
        # 0.25 phase remaining at 2.0 clip-seconds per wall second.
        self.assertAlmostEqual(times["mid"][0], 0.125, delta=0.015)
        # No event was replayed or back-dated by the schedule change:
        # every recorded 'mid' is a genuine new crossing.
        self.assertTrue(all(t > 0.1 for t in times["mid"]))

    def test_fired_events_survive_schedule_change(self):
        animator = _animator(self._library(), "gait")
        animator.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        animator.update(0.9)
        fired = [e["marker"] for e in animator.pop_events()]
        self.assertEqual(fired, ["mid"])
        animator.set_speed_schedule([(0.0, 1.0)])
        animator.update(0.9)
        markers = [e["marker"] for e in animator.pop_events()]
        # 'mid' of cycle 0 is not rewritten; only genuinely new
        # occurrences (foot_down of cycle 1, mid of cycle 1) appear.
        self.assertEqual(markers, ["foot_down", "mid"])

    def test_save_restore_mid_blend_continues_phase_and_schedule(self):
        library = self._library()
        library["other"] = _gait_clip("other", 1.0)
        states = [State("gait"), State("other")]
        continuous = CharacterAnimator(library, "other", states)
        restored = CharacterAnimator(library, "other", states)
        for animator in (continuous, restored):
            animator.request_transition("gait", 0.6)
            animator.set_speed_schedule(
                [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
            )
            animator.update(0.3)
        snapshot = restored.save_state()
        self.assertTrue(snapshot["player"]["speed_segments"])

        def run(animator):
            events = []
            for _ in range(120):
                animator.update(0.01)
                events.extend(e["marker"] for e in animator.pop_events())
            return events

        expected = run(continuous)
        restored.load_state(snapshot)
        actual = run(restored)
        self.assertEqual(expected, actual)
        self.assertAlmostEqual(
            continuous.player.state.local_time,
            restored.player.state.local_time,
            places=10,
        )
        self.assertAlmostEqual(
            continuous.root_position[0], restored.root_position[0], places=10
        )

    def test_reverse_with_schedule_fires_no_events_and_rewinds_root(self):
        animator = _animator(self._library(), "gait")
        animator.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=1.0
        )
        animator.update(0.9)  # clip time 0.5625, past 'mid'
        self.assertEqual(
            [e["marker"] for e in animator.pop_events()], ["mid"]
        )
        forward_x = animator.root_position[0]
        animator.update(-0.9)
        self.assertAlmostEqual(animator.player.state.local_time, 0.0)
        self.assertAlmostEqual(animator.root_position[0], 0.0)
        self.assertLess(animator.root_position[0], forward_x)
        self.assertEqual(animator.pop_events(), [])


class ScheduledThreeActionTests(unittest.TestCase):
    def test_third_switch_mid_blend_keeps_scheduled_phase(self):
        library = {
            "slow": _gait_clip("slow", 1.2),
            "fast": _gait_clip("fast", 0.6),
            "mid": _gait_clip("mid", 0.9),
        }
        animator = _animator(library, "slow")
        animator.update(0.72)
        animator.request_transition("fast", 0.4, sync_marker="foot_down")
        animator.set_speed_schedule(
            [(0.0, 0.5), (0.5, 2.0)], target_cycle_duration=0.6
        )
        animator.update(0.1)  # blend active, schedule active
        animator.request_transition("mid", 0.3, sync_marker="foot_down")
        entry = animator.transition_history()[-1]
        self.assertEqual(entry["source"], "fast")
        self.assertEqual(entry["destination"], "mid")
        self.assertAlmostEqual(
            entry["before"]["progress"], entry["after"]["progress"]
        )
        # Events still fire exactly once per cycle under the schedule.
        events = []
        for _ in range(200):
            animator.update(0.01)
            events.extend(animator.pop_events())
        keys = [(e["clip"], e["cycle"]) for e in events]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("mid", {e["clip"] for e in events})


if __name__ == "__main__":
    unittest.main()

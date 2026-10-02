import unittest

from animation_engine import (
    FootSpec,
    FootstepTracker,
    GroundSegment,
    Terrain,
    build_demo_library,
)
from animation_engine.player import AnimationPlayer


def make_feet(max_step_height=None, allowed_right=()):
    return [
        FootSpec("left", (0.0, 0.5), 0.3, max_step_height=max_step_height),
        FootSpec("right", (0.5, 1.0), 0.3,
                 allowed_segments=allowed_right,
                 max_step_height=max_step_height),
    ]


def make_tracker(terrain, feet=None, clip="walk"):
    library = build_demo_library()
    player = AnimationPlayer(library, clip)
    tracker = FootstepTracker(player, feet or make_feet(), terrain)
    return player, tracker


def flat_terrain():
    return Terrain([GroundSegment("flat", -10.0, 100.0, 0.0)])


def run(tracker, total, dt=0.05):
    steps = round(total / dt)
    for _ in range(steps):
        tracker.update(dt)


def assert_records_almost_equal(testcase, a, b):
    testcase.assertEqual(len(a), len(b))
    for ra, rb in zip(a, b):
        testcase.assertEqual(ra["foot"], rb["foot"])
        testcase.assertEqual(ra["clip"], rb["clip"])
        testcase.assertEqual(ra["reason"], rb["reason"])
        testcase.assertEqual(ra["status"], rb["status"])
        testcase.assertEqual(ra["segment_id"], rb["segment_id"])
        for key in ("plant_time", "lift_time", "slip"):
            if ra[key] is None or rb[key] is None:
                testcase.assertEqual(ra[key], rb[key])
            else:
                testcase.assertAlmostEqual(ra[key], rb[key], places=9)
        for key in ("position", "desired"):
            pa, pb = ra[key], rb[key]
            if pa is None or pb is None:
                testcase.assertEqual(pa, pb)
            else:
                testcase.assertAlmostEqual(pa[0], pb[0], places=9)
                testcase.assertAlmostEqual(pa[1], pb[1], places=9)


class AlternatingContactTest(unittest.TestCase):
    def test_alternating_contacts_and_root_progress(self):
        player, tracker = make_tracker(flat_terrain())
        run(tracker, 4.0)

        history = tracker.contact_history()
        self.assertGreaterEqual(len(history), 6)
        feet = [record["foot"] for record in history]
        # The very first left lift has no matching plant (the run starts
        # exactly on the marker), so closed records alternate right/left.
        self.assertEqual(feet[:4], ["right", "left", "right", "left"])
        for record in history:
            self.assertEqual(record["reason"], "ok")
            self.assertEqual(record["status"], "satisfied")
            self.assertEqual(record["segment_id"], "flat")
            self.assertAlmostEqual(record["slip"], 0.0)
            self.assertLess(record["plant_time"], record["lift_time"])

        # Root keeps advancing while feet hold their planted positions.
        self.assertGreater(player.root_position[0], 4.0)
        first_right = history[0]
        self.assertAlmostEqual(first_right["position"][0], 0.9, places=9)
        first_left = history[1]
        self.assertAlmostEqual(first_left["position"][0], 1.5, places=9)

    def test_planted_foot_position_is_held(self):
        _, tracker = make_tracker(flat_terrain())
        held = {"right": None}
        saw_plant = False
        for _ in range(40):  # 2.0 s
            tracker.update(0.05)
            report = tracker.contact_report()
            right = report["right"]
            if right["status"] == "planted":
                saw_plant = True
                if held["right"] is None:
                    held["right"] = right["position"]
                self.assertEqual(right["position"], held["right"])
            else:
                held["right"] = None
        self.assertTrue(saw_plant)


class UnsatisfiedContactTest(unittest.TestCase):
    def test_step_edge_too_high_reports_reason_and_slip(self):
        terrain = Terrain([
            GroundSegment("low", -10.0, 2.0, 0.0),
            GroundSegment("high", 2.0, 100.0, 0.5),
        ])
        feet = make_feet(max_step_height=0.2)
        player, tracker = make_tracker(terrain, feet)
        run(tracker, 3.0)

        history = tracker.contact_history()
        failed = [r for r in history if r["reason"] == "step_too_high"]
        self.assertTrue(failed)
        for record in failed:
            self.assertEqual(record["status"], "unsatisfied")
            self.assertIsNone(record["position"])
            self.assertIsNotNone(record["desired"])
            self.assertGreater(record["slip"], 0.0)
        # The character is never frozen to hide the failed plants.
        self.assertGreater(player.root_position[0], 3.0)
        satisfied = [r for r in history if r["reason"] == "ok"]
        self.assertTrue(satisfied)

    def test_segment_not_allowed(self):
        terrain = Terrain([
            GroundSegment("flat", -10.0, 100.0, 0.0),
            GroundSegment("pad", 5.0, 6.0, 0.0),
        ])
        feet = make_feet(allowed_right=("pad",))
        _, tracker = make_tracker(terrain, feet)
        run(tracker, 1.5)

        history = tracker.contact_history()
        right_records = [r for r in history if r["foot"] == "right"]
        self.assertTrue(right_records)
        for record in right_records:
            self.assertEqual(record["reason"], "segment_not_allowed")
            self.assertEqual(record["status"], "unsatisfied")
        left_records = [r for r in history if r["foot"] == "left"]
        self.assertTrue(all(r["reason"] == "ok" for r in left_records))


class RescheduleTest(unittest.TestCase):
    def test_speed_schedule_reschedules_future_plants_only(self):
        player, tracker = make_tracker(flat_terrain())
        run(tracker, 1.2)
        confirmed_before = [dict(r) for r in tracker.contact_history()]
        self.assertTrue(confirmed_before)

        # Slow the first half of the cycle to half speed, hurry the second.
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        run(tracker, 1.8)  # now at wall clock 3.0

        history = tracker.contact_history()
        # Already confirmed contacts are untouched by the new schedule.
        assert_records_almost_equal(
            self, confirmed_before, history[:len(confirmed_before)]
        )
        new_records = history[len(confirmed_before):]
        self.assertTrue(new_records)
        # Right foot plants at clip t=1.5; from t=1.2 the slow segment
        # (rate 0.5) stretches 0.3 clip-seconds into 0.6 wall seconds.
        right = next(r for r in new_records if r["foot"] == "right")
        self.assertAlmostEqual(right["plant_time"], 1.8, places=9)

    def test_target_duration_stretches_cycle(self):
        player, tracker = make_tracker(flat_terrain())
        run(tracker, 1.2)
        player.set_speed_schedule([(1.0, 1.0)], target_duration=2.0)
        run(tracker, 2.0)
        history = tracker.contact_history()
        right = next(
            r for r in history
            if r["foot"] == "right" and r["plant_time"] > 1.2
        )
        # Clip t=1.5 is 0.3 clip-seconds ahead of t=1.2; the 2 s target
        # halves the rate, so the plant lands 0.6 wall seconds later.
        self.assertAlmostEqual(right["plant_time"], 1.8, places=9)

    def test_reverse_playback_keeps_contacts(self):
        _, tracker = make_tracker(flat_terrain())
        run(tracker, 2.2)
        before = tracker.contact_history()
        report_before = tracker.contact_report()

        tracker.update(-0.5)  # rewind to t=1.7: no contacts lost or made
        self.assertEqual(len(tracker.contact_history()), len(before))
        tracker.update(0.5)  # forward again: no duplicates either
        assert_records_almost_equal(self, before, tracker.contact_history())
        report_after = tracker.contact_report()
        for name in report_before:
            self.assertEqual(
                report_before[name]["status"], report_after[name]["status"]
            )


class PersistenceTest(unittest.TestCase):
    def test_save_load_mid_stride_continues_each_foot(self):
        player, tracker = make_tracker(flat_terrain())
        run(tracker, 1.2)  # left planted (t in [1.0, 1.5)), right airborne
        report = tracker.contact_report()
        self.assertEqual(report["left"]["status"], "planted")
        self.assertEqual(report["right"]["status"], "airborne")

        saved_player = player.save_state()
        saved_tracker = tracker.save_state()

        # Reference: keep running the original.
        run(tracker, 2.8)

        # Restore into a fresh player+tracker and run the same span.
        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        player2.load_state(saved_player)
        tracker2 = FootstepTracker(player2, make_feet(), flat_terrain())
        tracker2.load_state(saved_tracker)
        restored_report = tracker2.contact_report()
        self.assertEqual(restored_report["left"]["status"], "planted")
        self.assertEqual(restored_report["right"]["status"], "airborne")
        run(tracker2, 2.8)

        assert_records_almost_equal(
            self, tracker.contact_history(), tracker2.contact_history()
        )

    def test_segmented_update_matches_single_replay(self):
        _, tracker_a = make_tracker(flat_terrain())
        run(tracker_a, 3.0, dt=0.05)

        _, tracker_b = make_tracker(flat_terrain())
        tracker_b.update(3.0)

        assert_records_almost_equal(
            self, tracker_a.contact_history(), tracker_b.contact_history()
        )

    def test_history_immune_to_later_terrain_edits(self):
        terrain = flat_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 2.0)
        before = [dict(r) for r in tracker.contact_history()]
        report_before = tracker.contact_report()
        self.assertTrue(before)

        # Edit the ground out from under the character afterwards.
        terrain.segments.clear()
        terrain.segments.append(GroundSegment("flat", -10.0, 100.0, 9.9))

        self.assertEqual(tracker.contact_history(), before)
        report_after = tracker.contact_report()
        for name in report_before:
            self.assertEqual(
                report_before[name]["height"], report_after[name]["height"]
            )
            self.assertEqual(
                report_before[name]["position"], report_after[name]["position"]
            )


class TransitionTest(unittest.TestCase):
    def test_third_clip_switch_mid_blend_keeps_feet_explainable(self):
        player, tracker = make_tracker(flat_terrain())
        run(tracker, 1.2)
        confirmed = [dict(r) for r in tracker.contact_history()]

        player.transition_to("run", duration=0.3, sync_marker="foot_down")
        run(tracker, 0.15)  # mid-blend: redirect into a third clip
        player.transition_to("walk", duration=0.2, sync_marker="foot_down")
        run(tracker, 2.0)

        history = tracker.contact_history()
        assert_records_almost_equal(
            self, confirmed, history[:len(confirmed)]
        )
        self.assertGreater(len(history), len(confirmed))
        report = tracker.contact_report()
        for name, foot in report.items():
            self.assertIn(foot["status"], ("airborne", "planted", "unsatisfied"))
            if foot["status"] == "planted":
                self.assertIsNotNone(foot["position"])
                self.assertEqual(foot["reason"], None)


class PlanningTest(unittest.TestCase):
    def gapped_terrain(self):
        return Terrain([
            GroundSegment("flat", -10.0, 3.0, 0.0),
            # hole between 3.0 and 3.4
            GroundSegment("pad", 3.4, 100.0, 0.0),
        ])

    def test_compare_keep_vs_adjust_over_hole(self):
        _, tracker = make_tracker(self.gapped_terrain())
        run(tracker, 1.2)

        plans = tracker.compare_plans(4)
        keep = plans["keep"]
        adjust = plans["adjust"]
        self.assertEqual(len(keep), 4)
        self.assertEqual(len(adjust), 4)

        keep_third = keep[2]
        self.assertEqual(keep_third["status"], "unsatisfied")
        self.assertEqual(keep_third["reason"], "no_ground")
        # Adjusting the stride length reaches the far pad instead.
        self.assertTrue(all(s["status"] == "satisfied" for s in adjust))
        self.assertNotEqual(
            adjust[2]["stride_offset"], keep[2]["stride_offset"]
        )
        # Planning is a simulation: live state is untouched.
        report = tracker.contact_report()
        self.assertEqual(report["left"]["status"], "planted")

    def test_unreachable_hole_stays_unsatisfied_in_both_plans(self):
        _, tracker = make_tracker(self.gapped_terrain())
        run(tracker, 1.2)
        plans = tracker.compare_plans(4, adjust_limit=0.05)
        self.assertEqual(plans["adjust"][2]["status"], "unsatisfied")
        self.assertEqual(plans["adjust"][2]["reason"], "no_ground")


if __name__ == "__main__":
    unittest.main()

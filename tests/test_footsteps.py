import unittest

from animation_engine import (
    FootSpec,
    GaitTracker,
    GroundSegment,
    Terrain,
    build_demo_library,
)
from animation_engine.player import AnimationPlayer


def make_tracker(terrain_segments=None, feet=None, clip="walk"):
    library = build_demo_library()
    player = AnimationPlayer(library, clip)
    if terrain_segments is None:
        terrain_segments = [GroundSegment(-10.0, 100.0, 0.0)]
    terrain = Terrain(terrain_segments)
    if feet is None:
        feet = [
            FootSpec("left", (0.0, 0.5), offset=(0.3, 0.1)),
            FootSpec("right", (0.5, 1.0), offset=(0.3, -0.1)),
        ]
    return GaitTracker(player, terrain, feet)


def run(tracker, total, dt):
    elapsed = 0.0
    while elapsed < total - 1e-12:
        step = min(dt, total - elapsed)
        tracker.update(step)
        elapsed += step


class AlternatingContactTest(unittest.TestCase):
    def test_alternating_records_and_root_progress(self):
        tracker = make_tracker()
        run(tracker, 3.0, 0.1)
        records = tracker.contact_records()
        self.assertEqual([r["foot"] for r in records],
                         ["left", "right", "left", "right", "left", "right"])
        # Plant times land exactly on window boundaries (cycle = 1 s).
        self.assertAlmostEqual(records[0]["plant_time"], 0.0)
        self.assertAlmostEqual(records[0]["lift_time"], 0.5)
        self.assertAlmostEqual(records[1]["plant_time"], 0.5)
        # Each confirmed plant keeps its own fixed ground position.
        xs = [r["position"][0] for r in records]
        self.assertTrue(all(b > a for a, b in zip(xs, xs[1:])))
        # Root keeps advancing while feet hold their positions.
        root_x = tracker.player.root_position[0]
        self.assertAlmostEqual(root_x, 3.6, places=6)
        # Slide is reported, not hidden: the root moved on during stance.
        self.assertTrue(all(r["max_slide"] > 0.0 for r in records))

    def test_foot_events_sequence(self):
        tracker = make_tracker()
        run(tracker, 1.0, 0.25)
        events = tracker.pop_foot_events()
        kinds = [(e["type"], e["foot"]) for e in events]
        self.assertEqual(
            kinds,
            [
                ("plant", "left"),
                ("liftoff", "left"),
                ("plant", "right"),
                ("liftoff", "right"),
                ("plant", "left"),
            ],
        )
        self.assertEqual(tracker.pop_foot_events(), [])

    def test_chunked_and_whole_updates_match(self):
        records_per_step = {}
        for dt in (0.05, 0.3, 2.5):
            tracker = make_tracker()
            run(tracker, 2.5, dt)
            records_per_step[dt] = tracker.contact_records()
        base = records_per_step[0.05]
        for dt in (0.3, 2.5):
            other = records_per_step[dt]
            self.assertEqual(len(base), len(other))
            for a, b in zip(base, other):
                self.assertEqual(a["foot"], b["foot"])
                self.assertAlmostEqual(a["plant_time"], b["plant_time"])
                self.assertAlmostEqual(a["lift_time"], b["lift_time"])
                self.assertAlmostEqual(a["position"][0], b["position"][0])


class TerrainTest(unittest.TestCase):
    def test_step_edge_heights(self):
        terrain = [
            GroundSegment(-10.0, 2.0, 0.0),
            GroundSegment(2.0, 100.0, 0.5),
        ]
        tracker = make_tracker(terrain_segments=terrain)
        run(tracker, 4.0, 0.1)
        records = tracker.contact_records()
        low = [r for r in records if r["position"][0] < 2.0]
        high = [r for r in records if r["position"][0] >= 2.0]
        self.assertTrue(low and high)
        self.assertTrue(all(r["position"][1] == 0.0 for r in low))
        self.assertTrue(all(r["position"][1] == 0.5 for r in high))

    def test_gap_reports_unsatisfied_with_reason(self):
        terrain = [GroundSegment(-10.0, 0.8, 0.0),
                   GroundSegment(1.6, 100.0, 0.0)]
        tracker = make_tracker(terrain_segments=terrain)
        run(tracker, 2.0, 0.05)
        records = tracker.contact_records()
        gaps = [r for r in records if not r["satisfied"]]
        self.assertTrue(gaps)
        self.assertTrue(all(r["reason"] == "no_ground" for r in gaps))
        self.assertTrue(all(r["position"] is None for r in gaps))
        # Unsatisfied contacts never lock the character: root still moves.
        self.assertGreater(tracker.player.root_position[0], 2.0)

    def test_disallowed_segment_reason(self):
        terrain = [GroundSegment(-10.0, 1.0, 0.0),
                   GroundSegment(1.0, 100.0, 0.0)]
        feet = [
            FootSpec("left", (0.0, 0.5), offset=(0.3, 0.1), allowed={0}),
            FootSpec("right", (0.5, 1.0), offset=(0.3, -0.1)),
        ]
        tracker = make_tracker(terrain_segments=terrain, feet=feet)
        run(tracker, 3.0, 0.05)
        left = [r for r in tracker.contact_records() if r["foot"] == "left"]
        self.assertTrue(any(
            not r["satisfied"] and r["reason"] == "segment_not_allowed"
            for r in left
        ))

    def test_terrain_edit_does_not_rewrite_history(self):
        terrain = Terrain([GroundSegment(-10.0, 100.0, 0.0)])
        library = build_demo_library()
        player = AnimationPlayer(library, "walk")
        feet = [FootSpec("left", (0.0, 0.5)), FootSpec("right", (0.5, 1.0))]
        tracker = GaitTracker(player, terrain, feet)
        run(tracker, 1.0, 0.1)
        before = tracker.contact_records()
        terrain.segments[0].height = 9.9
        terrain.add_segment(GroundSegment(200.0, 300.0, 1.0))
        after = tracker.contact_records()
        self.assertEqual(before, after)
        # New plants (confirmed after the edit) see the edited terrain.
        run(tracker, 1.0, 0.1)
        new_records = [
            r for r in tracker.contact_records()
            if r["plant_time"] > 1.0 + 1e-6
        ]
        self.assertTrue(new_records)
        self.assertTrue(all(
            r["position"][1] == 9.9 for r in new_records if r["satisfied"]
        ))


class SpeedChangeTest(unittest.TestCase):
    def test_future_plants_follow_new_schedule(self):
        tracker = make_tracker()
        tracker.update(0.25)
        left = tracker.contacts()["left"]
        self.assertEqual(left["status"], "planted")
        planted_x = left["position"][0]
        # Slow the stance half of the cycle down 4x from now on.
        tracker.player.set_speed_schedule([(0.5, 0.25), (1.0, 1.0)])
        tracker.update(1.75)
        records = tracker.contact_records()
        right = [r for r in records if r["foot"] == "right"]
        # Without the change the right foot would plant at wall t=0.5;
        # the remaining 0.25 cycle units at rate 0.25 take 1.0 s.
        self.assertAlmostEqual(right[0]["plant_time"], 1.25)
        # The confirmed left plant never moved.
        left_records = [r for r in records if r["foot"] == "left"]
        self.assertAlmostEqual(left_records[0]["position"][0], planted_x)
        self.assertAlmostEqual(left_records[0]["lift_time"], 1.25)

    def test_target_duration_stretches_cycle(self):
        tracker = make_tracker()
        tracker.player.set_speed_schedule(
            [(0.5, 1.0), (1.0, 1.0)], target_duration=2.0
        )
        tracker.update(0.0)
        run(tracker, 4.0, 0.2)
        records = tracker.contact_records()
        right = [r for r in records if r["foot"] == "right"]
        self.assertAlmostEqual(right[0]["plant_time"], 1.0)


class PersistenceTest(unittest.TestCase):
    def test_save_load_continues_each_foot(self):
        tracker = make_tracker()
        run(tracker, 0.75, 0.05)
        state = tracker.contacts()
        self.assertEqual(state["right"]["status"], "planted")
        self.assertEqual(state["left"]["status"], "airborne")
        saved = tracker.save_state()

        resumed = make_tracker()
        resumed.load_state(saved)
        state = resumed.contacts()
        self.assertEqual(state["right"]["status"], "planted")
        self.assertEqual(state["left"]["status"], "airborne")

        run(tracker, 1.25, 0.05)
        run(resumed, 1.25, 0.05)
        self.assertEqual(
            tracker.contact_records(), resumed.contact_records()
        )
        self.assertEqual(tracker.contacts(), resumed.contacts())
        self.assertAlmostEqual(
            tracker.player.root_position[0],
            resumed.player.root_position[0],
        )


class ReverseAndSwitchTest(unittest.TestCase):
    def test_reverse_keeps_foot_states(self):
        tracker = make_tracker()
        run(tracker, 0.75, 0.05)
        tracker.pop_foot_events()
        before = tracker.contacts()
        tracker.update(-0.3)
        self.assertEqual(tracker.pop_foot_events(), [])
        after = tracker.contacts()
        self.assertEqual(before["right"]["status"], after["right"]["status"])
        self.assertEqual(before["right"]["position"],
                         after["right"]["position"])

    def test_switch_mid_blend_keeps_contacts_explainable(self):
        tracker = make_tracker()
        run(tracker, 0.75, 0.05)  # right foot planted
        tracker.player.transition_to("run", 0.2, sync_marker="foot_down")
        tracker.update(0.1)
        tracker.player.transition_to("walk", 0.2, sync_marker="foot_down")
        run(tracker, 1.5, 0.05)
        records = tracker.contact_records()
        # The right-foot contact opened before the switches closes
        # normally at its window boundary; nothing teleports.
        right = [r for r in records if r["foot"] == "right"]
        self.assertTrue(right)
        self.assertIsNotNone(right[0]["lift_time"])
        for record in records:
            if record["satisfied"]:
                self.assertIsNotNone(record["position"])


class PlanComparisonTest(unittest.TestCase):
    def test_keep_gait_vs_adjust_stride(self):
        terrain = [GroundSegment(-10.0, 0.5, 0.0),
                   GroundSegment(1.5, 100.0, 0.0)]
        feet = [FootSpec("left", (0.0, 0.5), offset=(0.9, 0.0))]
        tracker = make_tracker(terrain_segments=terrain, feet=feet)
        plans = tracker.evaluate_plans("left")
        keep, adjust = plans
        # Nominal plant at x=0.9 sits over the gap.
        self.assertFalse(keep["satisfied"])
        self.assertEqual(keep["reason"], "no_ground")
        # Adjusting the stride pulls the plant back to the segment edge.
        self.assertTrue(adjust["satisfied"])
        self.assertAlmostEqual(adjust["position"][0], 0.5)
        self.assertAlmostEqual(adjust["stride_adjustment"], -0.4)
        # Evaluating plans commits nothing.
        self.assertEqual(tracker.contacts()["left"]["status"], "airborne")

    def test_unreachable_stays_unsatisfied(self):
        terrain = [GroundSegment(50.0, 100.0, 0.0)]
        feet = [FootSpec("left", (0.0, 0.5), offset=(0.3, 0.0),
                         max_reach=1.0)]
        tracker = make_tracker(terrain_segments=terrain, feet=feet)
        keep, adjust = tracker.evaluate_plans("left")
        self.assertFalse(keep["satisfied"])
        self.assertEqual(keep["reason"], "no_ground")
        self.assertFalse(adjust["satisfied"])
        self.assertEqual(adjust["reason"], "no_reachable_ground")


if __name__ == "__main__":
    unittest.main()

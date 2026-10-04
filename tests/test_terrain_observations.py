import unittest

from animation_engine import (
    FootSpec,
    FootstepTracker,
    GroundSegment,
    Terrain,
    build_demo_library,
)
from animation_engine.player import AnimationPlayer

from test_footsteps import assert_records_almost_equal


def make_feet():
    return [
        FootSpec("left", (0.0, 0.5), 0.3),
        FootSpec("right", (0.5, 1.0), 0.3),
    ]


def gapped_terrain():
    # Gap between x=2.0 and x=3.0; the walk cycle crosses it.
    return Terrain([
        GroundSegment("near", -10.0, 2.0, 0.0),
        GroundSegment("far", 3.0, 100.0, 0.0),
    ])


def make_tracker(terrain):
    library = build_demo_library()
    player = AnimationPlayer(library, "walk")
    tracker = FootstepTracker(player, make_feet(), terrain)
    return player, tracker


def run(tracker, total, dt=0.05):
    for _ in range(round(total / dt)):
        tracker.update(dt)


class ObservationVersionTest(unittest.TestCase):
    def test_register_and_retract_versions(self):
        terrain = gapped_terrain()
        self.assertEqual(terrain.current_version, 0)
        self.assertIsNone(terrain.height_at(2.5))

        v1 = terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.4)], note="survey"
        )
        self.assertEqual(v1, 1)
        self.assertEqual(terrain.height_at(2.5), 0.4)
        # Older versions remain reproducible.
        self.assertIsNone(terrain.height_at(2.5, version=0))

        v2 = terrain.register_observation([GroundSegment("deck", 2.0, 3.0, 0.9)])
        self.assertEqual(terrain.height_at(2.5), 0.9)
        self.assertEqual(terrain.height_at(2.5, version=v1), 0.4)

        terrain.retract_observation(v2)
        self.assertEqual(terrain.height_at(2.5), 0.4)
        with self.assertRaises(ValueError):
            terrain.retract_observation(v2)
        with self.assertRaises(ValueError):
            terrain.retract_observation(99)

    def test_observation_only_affects_future_by_default(self):
        terrain = gapped_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 2.6)
        before = [dict(r) for r in tracker.contact_history()]
        gaps = [r for r in before if r["reason"] == "no_ground"]
        self.assertTrue(gaps)
        for record in before:
            self.assertEqual(record["terrain_version"], 0)

        # Registering the fill-in platform does not rewrite history.
        terrain.register_observation([GroundSegment("deck", 2.0, 3.0, 0.4)])
        self.assertEqual(tracker.contact_history(), before)

        # Future plants are evaluated against the new version.
        run(tracker, 2.0)
        new_records = [
            r for r in tracker.contact_history() if r["plant_time"] > 2.6
        ]
        self.assertTrue(new_records)
        for record in new_records:
            self.assertEqual(record["terrain_version"], 1)


class ReinterpretTest(unittest.TestCase):
    def setUp(self):
        self.terrain = gapped_terrain()
        self.player, self.tracker = make_tracker(self.terrain)
        run(self.tracker, 3.0)
        self.gap_records = [
            r for r in self.tracker.contact_history()
            if r["reason"] == "no_ground"
        ]
        self.assertTrue(self.gap_records)

    def test_reinterpret_span_against_new_platform(self):
        version = self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.4)]
        )
        result = self.tracker.reinterpret(
            0.0, self.tracker.clock, version=version
        )
        self.assertEqual(result["reinterpreted"], len(self.gap_records))
        # Confirmed supports inside the span are frozen, never moved.
        self.assertGreaterEqual(result["frozen"], 1)

        # Original basis is preserved and still queryable.
        originals = {
            r["plant_time"]: r["reason"]
            for r in self.tracker.contact_history()
        }
        for record in self.gap_records:
            self.assertEqual(originals[record["plant_time"]], "no_ground")

        # The new explanation is queryable through interpreted_history.
        explained = [
            r for r in self.tracker.interpreted_history()
            if r["basis_reason"] == "no_ground"
        ]
        self.assertEqual(len(explained), len(self.gap_records))
        for view in explained:
            self.assertEqual(view["reason"], "ok")
            self.assertEqual(view["status"], "satisfied")
            self.assertEqual(view["segment_id"], "deck")
            self.assertAlmostEqual(view["height"], 0.4)
            self.assertEqual(view["terrain_version"], version)
            self.assertEqual(view["basis_terrain_version"], 0)

    def test_retract_restores_pending_records(self):
        version = self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.4)]
        )
        self.tracker.reinterpret(0.0, self.tracker.clock, version=version)

        untouched_before = self.tracker.interpreted_history()
        restored = self.tracker.retract_observation(version)
        self.assertEqual(restored, len(self.gap_records))

        views = self.tracker.interpreted_history()
        # Affected records fall back to the pending no_ground basis.
        for view in views:
            if view["plant_time"] in [r["plant_time"] for r in self.gap_records]:
                self.assertEqual(view["reason"], "no_ground")
                self.assertEqual(view["status"], "unsatisfied")
                self.assertIsNone(view["position"])
        # Unrelated records are unchanged by the retraction.
        for before, after in zip(untouched_before, views):
            if before["basis_reason"] == "ok":
                self.assertEqual(before, after)

    def test_frozen_supports_not_moved_by_height_revision(self):
        # Revise the height of ground a foot already stands on: the
        # confirmed plant keeps its original height and position.
        version = self.terrain.register_observation(
            [GroundSegment("near", -10.0, 2.0, 9.9)]
        )
        result = self.tracker.reinterpret(0.0, self.tracker.clock, version=version)
        self.assertEqual(result["reinterpreted"], len(self.gap_records))
        satisfied = [
            r for r in self.tracker.interpreted_history()
            if r["basis_reason"] == "ok"
        ]
        self.assertTrue(satisfied)
        for view in satisfied:
            self.assertEqual(view["reason"], "ok")
            self.assertAlmostEqual(view["height"], 0.0)


class DeterminismTest(unittest.TestCase):
    def scenario(self, rewind=False, snapshots=False):
        terrain = gapped_terrain()
        player, tracker = make_tracker(terrain)
        saved = []

        def checkpoint():
            if snapshots:
                saved.append((player.save_state(), tracker.save_state()))

        run(tracker, 1.6)
        checkpoint()
        # Mid-run survey: fill the gap with a platform at new height.
        terrain.register_observation([GroundSegment("deck", 2.0, 3.0, 0.4)])
        # Change speed mid-way: slow support, hurry swing.
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        checkpoint()
        run(tracker, 1.6)
        if rewind:
            tracker.update(-0.9)  # jump back over derived contacts
            tracker.update(0.9)   # forward again: no duplicates
        gaps = [
            r for r in tracker.contact_history() if r["reason"] == "no_ground"
        ]
        self.assertTrue(gaps)
        start = min(r["plant_time"] for r in gaps)
        end = max(r["plant_time"] for r in gaps)
        tracker.reinterpret(start, end)
        checkpoint()
        run(tracker, 1.4)
        return tracker, saved

    def test_rewind_and_forward_matches_continuous(self):
        continuous, _ = self.scenario()
        rewound, _ = self.scenario(rewind=True)
        assert_records_almost_equal(
            self, continuous.contact_history(), rewound.contact_history()
        )
        self.assertEqual(
            continuous.interpreted_history(), rewound.interpreted_history()
        )

    def test_save_load_between_events_matches_continuous(self):
        continuous, saved = self.scenario(snapshots=True)
        self.assertEqual(len(saved), 3)

        # Rebuild from the last checkpoint and finish the run.
        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        tracker2 = FootstepTracker(player2, make_feet(), gapped_terrain())
        player2.load_state(saved[-1][0])
        tracker2.load_state(saved[-1][1])
        run(tracker2, 1.4)

        assert_records_almost_equal(
            self, continuous.contact_history(), tracker2.contact_history()
        )
        self.assertEqual(
            continuous.interpreted_history(), tracker2.interpreted_history()
        )

    def test_reinterpretation_survives_save_load(self):
        _, saved = self.scenario(snapshots=True)
        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        tracker2 = FootstepTracker(player2, make_feet(), gapped_terrain())
        player2.load_state(saved[-1][0])
        tracker2.load_state(saved[-1][1])

        # The reloaded tracker still knows the observation and the
        # reinterpretation; retracting restores the pending records.
        explained = [
            r for r in tracker2.interpreted_history()
            if r["basis_reason"] == "no_ground" and r["reason"] == "ok"
        ]
        self.assertTrue(explained)
        tracker2.retract_observation(1)
        pending = [
            r for r in tracker2.interpreted_history()
            if r["reason"] == "no_ground"
        ]
        self.assertEqual(len(pending), len(explained))


class ConflictingObservationTest(unittest.TestCase):
    def test_select_conflicting_surveys_by_time_and_retract_only_dependency(self):
        terrain = gapped_terrain()
        player, tracker = make_tracker(terrain)
        run(tracker, 1.4)
        # Cross the gap while changing speed part-way through.
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        run(tracker, 1.6)

        gaps = [r for r in tracker.contact_history() if r["reason"] == "no_ground"]
        self.assertEqual(len(gaps), 2)
        first_time, second_time = [r["plant_time"] for r in gaps]

        low_version = terrain.register_observation(
            [GroundSegment("batch_a", 2.0, 2.5, 0.4)],
            note="low batch", source="survey-a",
        )
        high_version = terrain.register_observation(
            [GroundSegment("batch_b", 2.5, 3.0, 0.9)],
            note="high batch", source="survey-b",
        )

        comparison = tracker.compare_interpretations(
            first_time, second_time, [low_version, high_version]
        )
        self.assertEqual(comparison[low_version]["adopted"], 0)
        self.assertEqual(comparison[high_version]["adopted"], 0)

        # The newest registration does not automatically win.  Each time
        # span explicitly chooses the batch that actually covers it.
        self.assertEqual(
            tracker.select_interpretation(
                first_time, first_time, low_version
            )["selected"],
            1,
        )
        self.assertEqual(
            tracker.select_interpretation(
                second_time, second_time, high_version
            )["selected"],
            1,
        )

        views_by_time = {
            view["plant_time"]: view
            for view in tracker.interpreted_history()
            if view["basis_reason"] == "no_ground"
        }
        first = views_by_time[first_time]
        second = views_by_time[second_time]
        self.assertEqual(first["segment_id"], "batch_a")
        self.assertEqual(first["source"], "survey-a")
        self.assertAlmostEqual(first["height"], 0.4)
        self.assertEqual(first["selected_terrain_version"], low_version)
        self.assertEqual(second["segment_id"], "batch_b")
        self.assertEqual(second["source"], "survey-b")
        self.assertAlmostEqual(second["height"], 0.9)
        self.assertEqual(second["selected_terrain_version"], high_version)

        # Withdraw the low batch: only its dependent explanation is removed.
        # The first point is pending again, while the high batch survives.
        restored = tracker.retract_observation(low_version)
        self.assertEqual(restored, 1)
        views_by_time = {
            view["plant_time"]: view
            for view in tracker.interpreted_history()
            if view["basis_reason"] == "no_ground"
        }
        self.assertEqual(views_by_time[first_time]["reason"], "no_ground")
        self.assertIsNone(views_by_time[first_time]["position"])
        self.assertEqual(views_by_time[second_time]["reason"], "ok")
        self.assertEqual(views_by_time[second_time]["source"], "survey-b")

    def test_unresolved_conflicts_survive_reload_rewind_and_forward(self):
        def build(rewind_after_load=False):
            terrain = gapped_terrain()
            player, tracker = make_tracker(terrain)
            run(tracker, 1.4)
            player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
            run(tracker, 1.6)
            gaps = [
                record for record in tracker.contact_history()
                if record["reason"] == "no_ground"
            ]
            first_time, second_time = [r["plant_time"] for r in gaps]
            low_version = terrain.register_observation(
                [GroundSegment("batch_a", 2.0, 2.5, 0.4)],
                source="survey-a",
            )
            high_version = terrain.register_observation(
                [GroundSegment("batch_b", 2.5, 3.0, 0.9)],
                source="survey-b",
            )
            tracker.compare_interpretations(
                first_time, second_time, [low_version, high_version]
            )
            saved_player = player.save_state()
            saved_tracker = tracker.save_state()
            if rewind_after_load:
                player2 = AnimationPlayer(build_demo_library(), "walk")
                tracker2 = FootstepTracker(
                    player2, make_feet(), gapped_terrain()
                )
                player2.load_state(saved_player)
                tracker2.load_state(saved_tracker)
                tracker2.update(-0.9)
                tracker2.update(0.9)
                player, tracker = player2, tracker2
            tracker.select_interpretation(
                first_time, first_time, low_version
            )
            tracker.select_interpretation(
                second_time, second_time, high_version
            )
            return tracker

        continuous = build()
        rewound = build(rewind_after_load=True)
        self.assertEqual(
            len(continuous.contact_history()),
            len(rewound.contact_history()),
        )
        assert_records_almost_equal(
            self,
            continuous.contact_history(),
            rewound.contact_history(),
        )
        self.assertEqual(
            continuous.interpreted_history(),
            rewound.interpreted_history(),
        )


if __name__ == "__main__":
    unittest.main()

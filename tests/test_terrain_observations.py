import unittest

from animation_engine import (
    FootSpec,
    FootstepTracker,
    GroundSegment,
    Terrain,
    build_demo_library,
)
from animation_engine.player import AnimationPlayer

from test_footsteps import assert_records_almost_equal, make_feet, run


def make_tracker(terrain, feet=None, clip="walk"):
    library = build_demo_library()
    player = AnimationPlayer(library, clip)
    tracker = FootstepTracker(player, feet or make_feet(), terrain)
    return player, tracker


def gap_terrain():
    # A hole between x=1.5 and x=2.5 that the walk cycle crosses.
    return Terrain([
        GroundSegment("near", -10.0, 1.5, 0.0),
        GroundSegment("far", 3.5, 100.0, 0.0),
    ])


def deck_survey(height=0.4):
    # Later survey: the gap is actually bridged by a platform.
    return [
        GroundSegment("near", -10.0, 1.5, 0.0),
        GroundSegment("deck", 1.5, 3.5, height),
        GroundSegment("far", 3.5, 100.0, 0.0),
    ]


def gap_records(tracker):
    return [
        record for record in tracker.contact_history()
        if record["reason"] == "no_ground"
    ]


class VersionedTerrainTest(unittest.TestCase):
    def test_observations_are_versioned_and_retractable(self):
        terrain = gap_terrain()
        self.assertEqual(terrain.version, 1)
        self.assertIsNone(terrain.height_at(2.0))

        version = terrain.register_observation(deck_survey())
        self.assertEqual(version, 2)
        self.assertEqual(terrain.version, 2)
        self.assertAlmostEqual(terrain.height_at(2.0), 0.4)

        terrain.retract_observation(version)
        self.assertEqual(terrain.version, 1)
        self.assertIsNone(terrain.height_at(2.0))

    def test_terrain_state_round_trips(self):
        terrain = gap_terrain()
        terrain.register_observation(deck_survey(), note="resurvey")
        terrain.retract_observation(1)
        restored = Terrain([])
        restored.load_state(terrain.save_state())
        self.assertEqual(restored.version, 2)
        self.assertAlmostEqual(restored.height_at(2.0), 0.4)
        self.assertIsNone(restored.height_at(-20.0))


class FutureOnlyObservationTest(unittest.TestCase):
    def test_future_mode_keeps_history_and_speed_change_is_consistent(self):
        terrain = gap_terrain()
        player, tracker = make_tracker(terrain)
        run(tracker, 2.2)
        before = [dict(record) for record in tracker.contact_history()]
        self.assertTrue(gap_records(tracker))

        # Mid-run speed change: only reschedules what has not happened.
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        version = tracker.register_observation(deck_survey(), mode="future")
        self.assertEqual(version, 2)

        # History is untouched: same records, same basis version.
        assert_records_almost_equal(self, before, tracker.contact_history())
        for record in tracker.contact_history():
            self.assertEqual(record["terrain_version"], 1)

        # Future plants inside the old gap now land on the deck.
        run(tracker, 2.0)
        later = tracker.contact_history()[len(before):]
        self.assertTrue(later)
        decked = [r for r in later if r["segment_id"] == "deck"]
        self.assertTrue(decked)
        for record in decked:
            self.assertEqual(record["reason"], "ok")
            self.assertAlmostEqual(record["height"], 0.4)
            self.assertEqual(record["terrain_version"], 2)


class ReinterpretationTest(unittest.TestCase):
    def test_reinterpret_backfills_gap_and_keeps_originals_queryable(self):
        terrain = gap_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 3.0)
        pending = gap_records(tracker)
        self.assertTrue(pending)
        since = min(r["plant_time"] for r in pending) - 1e-6
        until = max(r["plant_time"] for r in pending) + 1e-6

        tracker.register_observation(
            deck_survey(), mode="reinterpret", since=since, until=until
        )

        # Effective history: the gap plants now read as deck landings.
        effective = tracker.contact_history()
        self.assertFalse(gap_records(tracker))
        decked = [r for r in effective if r["segment_id"] == "deck"]
        self.assertEqual(len(decked), len(pending))
        for record in decked:
            self.assertEqual(record["status"], "satisfied")
            self.assertEqual(record["terrain_version"], 2)
            self.assertIsNotNone(record["reinterpreted_from"])

        # The original no_ground records keep their basis and stay
        # queryable alongside the new interpretation.
        full = tracker.contact_history(include_superseded=True)
        self.assertEqual(len(full), len(effective) + len(pending))
        originals = [r for r in full if r["superseded_by"] is not None]
        self.assertEqual(len(originals), len(pending))
        for record in originals:
            self.assertEqual(record["reason"], "no_ground")
            self.assertEqual(record["terrain_version"], 1)
            self.assertIsNone(record["position"])

    def test_reinterpret_span_limits_which_records_change(self):
        terrain = gap_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 3.0)
        pending = gap_records(tracker)
        self.assertGreaterEqual(len(pending), 2)
        first = pending[0]

        # Reinterpret only the earliest gap plant.
        tracker.register_observation(
            deck_survey(),
            mode="reinterpret",
            since=first["plant_time"] - 1e-6,
            until=first["plant_time"] + 1e-6,
        )
        remaining = gap_records(tracker)
        self.assertEqual(len(remaining), len(pending) - 1)

    def test_frozen_support_is_never_moved(self):
        terrain = gap_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 1.2)  # left foot planted on "near"
        report = tracker.contact_report()
        self.assertEqual(report["left"]["status"], "planted")
        planted_position = report["left"]["position"]

        tracker.register_observation(
            deck_survey(height=9.9), mode="reinterpret"
        )
        report_after = tracker.contact_report()
        self.assertEqual(report_after["left"]["status"], "planted")
        self.assertEqual(report_after["left"]["position"], planted_position)
        self.assertEqual(report_after["left"]["segment_id"], "near")

        # Satisfied history is equally frozen.
        run(tracker, 1.8)
        for record in tracker.contact_history():
            if record["status"] == "satisfied":
                self.assertIsNone(record["reinterpreted_from"])


class RetractionTest(unittest.TestCase):
    def test_retract_restores_pending_without_touching_unrelated(self):
        terrain = gap_terrain()
        _, tracker = make_tracker(terrain)
        run(tracker, 3.0)
        pending_before = gap_records(tracker)
        satisfied_before = [
            r for r in tracker.contact_history() if r["status"] == "satisfied"
        ]
        tracker.register_observation(deck_survey(), mode="reinterpret")
        self.assertFalse(gap_records(tracker))

        tracker.retract_observation(2)

        # The gap plants are pending again with their original basis.
        restored = gap_records(tracker)
        self.assertEqual(len(restored), len(pending_before))
        for record in restored:
            self.assertEqual(record["status"], "unsatisfied")
            self.assertEqual(record["terrain_version"], 1)
            self.assertIsNone(record["superseded_by"])
        # Unrelated gait is byte-identical.
        satisfied_after = [
            r for r in tracker.contact_history() if r["status"] == "satisfied"
        ]
        assert_records_almost_equal(self, satisfied_before, satisfied_after)
        # And the terrain itself fell back to the pre-survey profile.
        self.assertIsNone(terrain.height_at(2.0))


class DeterminismTest(unittest.TestCase):
    def _scenario(self, save_midway=False, rewind=False):
        terrain = gap_terrain()
        player, tracker = make_tracker(terrain)
        run(tracker, 1.0)
        tracker.register_observation(deck_survey(), mode="future")
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        if rewind:
            tracker.update(-0.4)
            tracker.update(0.4)
        if save_midway:
            saved = (
                player.save_state(),
                tracker.save_state(),
                terrain.save_state(),
            )
            run(tracker, 0.6)
            player2 = AnimationPlayer(build_demo_library(), "walk")
            player2.load_state(saved[0])
            terrain2 = Terrain([])
            terrain2.load_state(saved[2])
            tracker2 = FootstepTracker(player2, make_feet(), terrain2)
            tracker2.load_state(saved[1])
            tracker = tracker2
        run(tracker, 2.4)
        return tracker.contact_history()

    def test_continuous_reload_and_rewind_agree(self):
        reference = self._scenario()
        assert_records_almost_equal(
            self, reference, self._scenario(save_midway=True)
        )
        assert_records_almost_equal(
            self, reference, self._scenario(rewind=True)
        )


if __name__ == "__main__":
    unittest.main()

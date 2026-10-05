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



def assert_interpreted_almost_equal(testcase, views_a, views_b):
    testcase.assertEqual(len(views_a), len(views_b))
    for va, vb in zip(views_a, views_b):
        testcase.assertEqual(set(va), set(vb))
        for key in va:
            xa, xb = va[key], vb[key]
            if xa is None or xb is None:
                testcase.assertEqual(xa, xb)
            elif isinstance(xa, float):
                testcase.assertAlmostEqual(xa, xb, places=9)
            elif isinstance(xa, tuple):
                testcase.assertEqual(len(xa), len(xb))
                for ya, yb in zip(xa, xb):
                    testcase.assertAlmostEqual(ya, yb, places=9)
            else:
                testcase.assertEqual(xa, xb)


class ConflictingBatchesTest(unittest.TestCase):
    """Two contradictory survey batches are kept side by side; each time
    span of history picks which batch explains it."""

    def setUp(self):
        self.terrain = gapped_terrain()
        self.player, self.tracker = make_tracker(self.terrain)
        run(self.tracker, 3.0)
        self.gap_records = [
            r for r in self.tracker.contact_history()
            if r["reason"] == "no_ground"
        ]
        self.assertGreaterEqual(len(self.gap_records), 2)
        # Two conflicting surveys of the same gap: different heights and
        # different coverage, registered as separate batches.
        self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.4)], batch="survey_a"
        )
        self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.9),
             GroundSegment("far", 3.0, 100.0, 0.9)],
            batch="survey_b",
        )

    def split_point(self):
        times = sorted(r["plant_time"] for r in self.gap_records)
        k = len(times) // 2
        return (times[k - 1] + times[k]) / 2.0

    def test_conflicting_batches_coexist(self):
        # The latest registration wins in the fully stacked default view.
        self.assertEqual(self.terrain.height_at(2.5), 0.9)
        # Each batch stays selectable on its own, conflict preserved.
        self.assertEqual(self.terrain.height_at(2.5, batch="survey_a"), 0.4)
        self.assertEqual(self.terrain.height_at(2.5, batch="survey_b"), 0.9)
        self.assertEqual(self.terrain.height_at(3.5, batch="survey_a"), 0.0)
        self.assertEqual(self.terrain.height_at(3.5, batch="survey_b"), 0.9)
        self.assertEqual(self.terrain.batch_versions("survey_a"), [1])
        self.assertEqual(self.terrain.batch_versions("survey_b"), [2])

    def test_select_batch_per_time_span(self):
        mid = self.split_point()
        clock = self.tracker.clock
        result_a = self.tracker.reinterpret(0.0, mid, batch="survey_a")
        result_b = self.tracker.reinterpret(mid, clock, batch="survey_b")
        self.assertGreater(result_a["reinterpreted"], 0)
        self.assertGreater(result_b["reinterpreted"], 0)
        # Confirmed supports are frozen in both passes, never moved.
        self.assertGreater(result_a["frozen"], 0)
        self.assertGreater(result_b["frozen"], 0)

        views = {
            r["plant_time"]: r for r in self.tracker.interpreted_history()
        }
        for record in self.gap_records:
            view = views[record["plant_time"]]
            self.assertEqual(view["basis_reason"], "no_ground")
            self.assertEqual(view["basis_terrain_version"], 0)
            if record["plant_time"] <= mid:
                # Earlier span explained by survey_a only.
                self.assertEqual(view["batch"], "survey_a")
                self.assertEqual(view["reason"], "ok")
                self.assertAlmostEqual(view["height"], 0.4)
            else:
                # Later span explained by survey_b only.
                self.assertEqual(view["batch"], "survey_b")
                self.assertEqual(view["reason"], "ok")
                self.assertAlmostEqual(view["height"], 0.9)

    def test_latest_registration_does_not_override_confirmed_support(self):
        # A newer batch revising ground a foot already stands on must not
        # rewrite the confirmed support.
        self.terrain.register_observation(
            [GroundSegment("near", -10.0, 2.0, 9.9)], batch="survey_c"
        )
        result = self.tracker.reinterpret(
            0.0, self.tracker.clock, batch="survey_c"
        )
        satisfied = [
            r for r in self.tracker.interpreted_history()
            if r["basis_reason"] == "ok"
        ]
        self.assertTrue(satisfied)
        self.assertEqual(result["frozen"], len(satisfied))
        for view in satisfied:
            self.assertEqual(view["reason"], "ok")
            self.assertAlmostEqual(view["height"], 0.0)
        # The new batch says nothing about the gap: those records stay
        # pending instead of being force-explained.
        for record in self.gap_records:
            view = [
                r for r in self.tracker.interpreted_history()
                if r["plant_time"] == record["plant_time"]
            ][0]
            self.assertEqual(view["reason"], "no_ground")
            self.assertEqual(view["status"], "unsatisfied")

    def test_partially_observed_positions_stay_pending(self):
        # A batch covering only part of the gap explains only what it
        # observes; the rest keeps its pending no_ground state.
        self.terrain.register_observation(
            [GroundSegment("ledge", 2.0, 2.4, 0.3)], batch="partial"
        )
        self.tracker.reinterpret(0.0, self.tracker.clock, batch="partial")
        views = [
            r for r in self.tracker.interpreted_history()
            if r["basis_reason"] == "no_ground"
        ]
        explained = [v for v in views if v["reason"] == "ok"]
        pending = [v for v in views if v["reason"] == "no_ground"]
        self.assertTrue(explained)
        self.assertTrue(pending)
        for view in pending:
            self.assertEqual(view["status"], "unsatisfied")
            self.assertIsNone(view["position"])
            self.assertEqual(view["batch"], "partial")

    def test_retract_batch_restores_only_dependents(self):
        mid = self.split_point()
        clock = self.tracker.clock
        self.tracker.reinterpret(0.0, mid, batch="survey_a")
        self.tracker.reinterpret(mid, clock, batch="survey_b")
        before = self.tracker.interpreted_history()

        restored = self.tracker.retract_batch("survey_b")
        self.assertGreater(restored, 0)
        # Batch b is gone from the terrain; batch a still applies.
        self.assertEqual(self.terrain.height_at(2.5), 0.4)
        with self.assertRaises(ValueError):
            self.tracker.retract_batch("survey_b")

        views = self.tracker.interpreted_history()
        for old, view in zip(before, views):
            if old["basis_reason"] == "ok":
                # Confirmed supports were never touched.
                self.assertEqual(old, view)
            elif old["plant_time"] <= mid:
                # survey_a explanations survive the retraction of b.
                self.assertEqual(view["batch"], "survey_a")
                self.assertEqual(view["reason"], "ok")
                self.assertAlmostEqual(view["height"], 0.4)
            else:
                # Dependents of survey_b fall back to pending.
                self.assertEqual(view["reason"], "no_ground")
                self.assertEqual(view["status"], "unsatisfied")
                self.assertIsNone(view["batch"])

    def test_retract_single_version_keeps_sibling_batch(self):
        mid = self.split_point()
        clock = self.tracker.clock
        self.tracker.reinterpret(0.0, mid, batch="survey_a")
        self.tracker.reinterpret(mid, clock, batch="survey_b")
        # Retracting only version 1 (survey_a) leaves survey_b choices.
        self.tracker.retract_observation(1)
        views = self.tracker.interpreted_history()
        for view in views:
            if view["basis_reason"] != "no_ground":
                continue
            if view["plant_time"] <= mid:
                self.assertEqual(view["reason"], "no_ground")
            else:
                self.assertEqual(view["batch"], "survey_b")
                self.assertAlmostEqual(view["height"], 0.9)


class BatchPersistenceTest(unittest.TestCase):
    """Pending conflicts survive save/load; rewind and replay stay
    identical to continuous playback in count, source and outcome."""


class PartialWithdrawalTest(unittest.TestCase):
    """A reviewer withdraws the adoption decision for one span only,
    without retracting the survey batch behind the other span."""

    def setUp(self):
        self.terrain = gapped_terrain()
        self.player, self.tracker = make_tracker(self.terrain)
        run(self.tracker, 3.0)
        self.gap_records = [
            r for r in self.tracker.contact_history()
            if r["reason"] == "no_ground"
        ]
        self.assertGreaterEqual(len(self.gap_records), 2)
        # One resurvey batch covers two separate landing spans, with the
        # latest version numbering each record adopts.
        self.version = self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.4)], batch="resurvey"
        )
        times = sorted(r["plant_time"] for r in self.gap_records)
        k = len(times) // 2
        self.mid = (times[k - 1] + times[k]) / 2
        self.earlier = [t for t in times if t <= self.mid]
        self.later = [t for t in times if t > self.mid]
        self.assertTrue(self.earlier)
        self.assertTrue(self.later)
        self.tracker.reinterpret(
            0.0, self.mid, batch="resurvey",
            reviewer="alice", note="deck fills the first gap",
        )
        self.tracker.reinterpret(
            self.mid, self.tracker.clock, batch="resurvey",
            reviewer="alice", note="same deck carries the second gap",
        )

    def _views(self):
        return {
            r["plant_time"]: r
            for r in self.tracker.interpreted_history()
        }

    def test_withdraw_one_span_keeps_sibling_confirmation(self):
        result = self.tracker.withdraw_interpretation(
            0.0, self.mid, reviewer="bob", note="alice misread the span"
        )
        self.assertEqual(result["withdrawn"], len(self.earlier))

        # The resurvey batch itself is still fully registered: nothing
        # was retracted from the terrain.
        self.assertEqual(
            self.terrain.batch_versions("resurvey"), [self.version]
        )
        self.assertEqual(
            self.terrain.height_at(2.5, batch="resurvey"), 0.4
        )

        views = self._views()
        for plant_time in self.earlier:
            view = views[plant_time]
            # Withdrawn span falls back to its original pending basis;
            # no newer observation is auto-applied.
            self.assertEqual(view["reason"], "no_ground")
            self.assertEqual(view["status"], "unsatisfied")
            self.assertIsNone(view["position"])
            self.assertIsNone(view["batch"])
            self.assertEqual(view["basis_terrain_version"], 0)
        for plant_time in self.later:
            view = views[plant_time]
            # The other span's confirmation survives untouched.
            self.assertEqual(view["batch"], "resurvey")
            self.assertEqual(view["terrain_version"], self.version)
            self.assertEqual(view["reason"], "ok")
            self.assertEqual(view["reviewer"], "alice")
            self.assertEqual(view["note"], "same deck carries the second gap")
            self.assertAlmostEqual(view["height"], 0.4)

        # The withdrawn decision itself is preserved with its operator
        # note and the observation version it relied on.
        audit = {
            item["plant_time"]: item for item in self.tracker.decision_audit()
        }
        for plant_time in self.earlier:
            decisions = audit[plant_time]["decisions"]
            self.assertEqual(len(decisions), 1)
            decision = decisions[0]
            self.assertTrue(decision["withdrawn"])
            self.assertEqual(decision["reviewer"], "alice")
            self.assertEqual(decision["note"], "deck fills the first gap")
            self.assertEqual(decision["terrain_version"], self.version)
            self.assertEqual(decision["batch"], "resurvey")
            self.assertEqual(decision["withdrawn"]["reviewer"], "bob")
            self.assertEqual(
                decision["withdrawn"]["note"], "alice misread the span"
            )
        for plant_time in self.later:
            self.assertFalse(audit[plant_time]["decisions"][0]["withdrawn"])

    def test_conflicting_observation_does_not_auto_support_withdrawn_span(self):
        self.tracker.withdraw_interpretation(
            0.0, self.mid, reviewer="bob", note="span disputed"
        )
        # Registering a contradictory survey must not silently re-explain
        # the withdrawn span.
        self.terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 9.9)], batch="contradiction"
        )
        views = self._views()
        for plant_time in self.earlier:
            self.assertEqual(views[plant_time]["reason"], "no_ground")
            self.assertIsNone(views[plant_time]["batch"])
        for plant_time in self.later:
            # Still the resurvey confirmation, not the new height.
            self.assertEqual(views[plant_time]["batch"], "resurvey")
            self.assertAlmostEqual(views[plant_time]["height"], 0.4)

    def test_reconfirm_after_withdrawal_is_separate_decision(self):
        self.tracker.withdraw_interpretation(
            0.0, self.mid, reviewer="bob", note="re-survey needed"
        )
        self.tracker.reinterpret(
            0.0, self.mid, batch="resurvey",
            reviewer="carol", note="rechecked, deck really is there",
        )
        views = self._views()
        for plant_time in self.earlier:
            view = views[plant_time]
            self.assertEqual(view["reason"], "ok")
            self.assertEqual(view["batch"], "resurvey")
            self.assertEqual(view["reviewer"], "carol")

        audit = {
            item["plant_time"]: item for item in self.tracker.decision_audit()
        }
        for plant_time in self.earlier:
            decisions = audit[plant_time]["decisions"]
            self.assertEqual(len(decisions), 2)
            self.assertTrue(decisions[0]["withdrawn"])
            self.assertEqual(decisions[0]["reviewer"], "alice")
            self.assertFalse(decisions[1]["withdrawn"])
            self.assertEqual(decisions[1]["reviewer"], "carol")
            self.assertEqual(
                decisions[1]["terrain_version"], self.version
            )

    def test_withdrawal_survives_save_reload_and_rewind_replay(self):
        self.tracker.withdraw_interpretation(
            0.0, self.mid, reviewer="bob", note="span disputed"
        )
        saved_player = self.player.save_state()
        saved_tracker = self.tracker.save_state()

        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        tracker2 = FootstepTracker(player2, make_feet(), self.terrain)
        player2.load_state(saved_player)
        tracker2.load_state(saved_tracker)

        # Rewind over contacts and play forward again: no duplicate
        # decisions and the withdrawn span stays pending while the other
        # span stays confirmed.
        tracker2.update(-0.9)
        tracker2.update(0.9)
        assert_interpreted_almost_equal(
            self,
            self.tracker.interpreted_history(),
            tracker2.interpreted_history(),
        )
        audit_a = self.tracker.decision_audit()
        audit_b = tracker2.decision_audit()
        self.assertEqual(
            [len(item["decisions"]) for item in audit_a],
            [len(item["decisions"]) for item in audit_b],
        )
        for item_a, item_b in zip(audit_a, audit_b):
            self.assertEqual(item_a["decisions"], item_b["decisions"])

        # The sibling span can still be withdrawn independently after
        # reload, and the resurvey batch is never auto-retracted.
        result = tracker2.withdraw_interpretation(
            self.mid, tracker2.clock, reviewer="bob", note="other span out"
        )
        self.assertEqual(result["withdrawn"], len(self.later))
        self.assertEqual(
            self.terrain.batch_versions("resurvey"), [self.version]
        )
        for view in tracker2.interpreted_history():
            if view["basis_reason"] == "no_ground":
                self.assertEqual(view["reason"], "no_ground")

    def scenario(self, rewind=False, snapshots=False):
        terrain = gapped_terrain()
        player, tracker = make_tracker(terrain)
        saved = []

        def checkpoint():
            if snapshots:
                saved.append((player.save_state(), tracker.save_state()))

        run(tracker, 1.6)
        checkpoint()
        # First survey batch (partial coverage of the gap), then a
        # mid-run speed change.
        terrain.register_observation(
            [GroundSegment("deck", 2.0, 2.5, 0.4)], batch="survey_a"
        )
        player.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)])
        run(tracker, 1.6)
        checkpoint()
        # A second, conflicting batch registered later.
        terrain.register_observation(
            [GroundSegment("deck", 2.0, 3.0, 0.9)], batch="survey_b"
        )
        run(tracker, 1.0)
        if rewind:
            tracker.update(-0.9)  # back over derived contacts
            tracker.update(0.9)   # forward again: no duplicates
        gaps = [
            r for r in tracker.contact_history() if r["reason"] == "no_ground"
        ]
        self.assertTrue(gaps)
        times = sorted(r["plant_time"] for r in gaps)
        self.assertGreaterEqual(len(times), 2)
        k = len(times) // 2
        mid = (times[k - 1] + times[k]) / 2.0
        tracker.reinterpret(0.0, mid, batch="survey_a")
        tracker.reinterpret(mid, tracker.clock, batch="survey_b")
        checkpoint()
        run(tracker, 1.2)
        return tracker, saved

    def sources(self, tracker):
        return [
            (r["plant_time"], r["foot"], r["reason"], r["batch"])
            for r in tracker.interpreted_history()
        ]

    def test_rewind_and_forward_matches_continuous(self):
        continuous, _ = self.scenario()
        rewound, _ = self.scenario(rewind=True)
        self.assertEqual(
            len(continuous.contact_history()), len(rewound.contact_history())
        )
        assert_records_almost_equal(
            self, continuous.contact_history(), rewound.contact_history()
        )
        assert_interpreted_almost_equal(
            self, continuous.interpreted_history(), rewound.interpreted_history()
        )

    def test_save_load_with_pending_conflicts_matches_continuous(self):
        continuous, saved = self.scenario(snapshots=True)
        self.assertEqual(len(saved), 3)

        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        tracker2 = FootstepTracker(player2, make_feet(), gapped_terrain())
        player2.load_state(saved[-1][0])
        tracker2.load_state(saved[-1][1])
        run(tracker2, 1.2)

        self.assertEqual(
            len(continuous.contact_history()), len(tracker2.contact_history())
        )
        assert_records_almost_equal(
            self, continuous.contact_history(), tracker2.contact_history()
        )
        assert_interpreted_almost_equal(
            self, continuous.interpreted_history(), tracker2.interpreted_history()
        )
        self.assertEqual(self.sources(continuous), self.sources(tracker2))

    def test_reloaded_batches_remain_retractable(self):
        _, saved = self.scenario(snapshots=True)
        library = build_demo_library()
        player2 = AnimationPlayer(library, "walk")
        tracker2 = FootstepTracker(player2, make_feet(), gapped_terrain())
        player2.load_state(saved[-1][0])
        tracker2.load_state(saved[-1][1])

        views_before = tracker2.interpreted_history()
        explained_b = [
            r for r in views_before
            if r.get("batch") == "survey_b" and r["reason"] == "ok"
        ]
        explained_a = [
            r for r in views_before
            if r.get("batch") == "survey_a" and r["reason"] == "ok"
        ]
        pending_before = [
            r for r in views_before if r["reason"] == "no_ground"
        ]
        self.assertTrue(explained_a)
        self.assertTrue(explained_b)
        restored = tracker2.retract_batch("survey_b")
        self.assertEqual(restored, len(explained_b))
        still_a = [
            r for r in tracker2.interpreted_history()
            if r.get("batch") == "survey_a" and r["reason"] == "ok"
        ]
        self.assertEqual(len(still_a), len(explained_a))
        pending = [
            r for r in tracker2.interpreted_history()
            if r["reason"] == "no_ground"
        ]
        self.assertEqual(len(pending), len(pending_before) + len(explained_b))


if __name__ == "__main__":
    unittest.main()

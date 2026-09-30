import unittest

from animation_engine import (
    CharacterAnimator,
    State,
    build_demo_library,
)


STATES = [State("walk"), State("run"), State("idle"), State("stride", "walk")]


def _scenario(animator, advance):
    """walk -> run -> idle while the run blend is still active.

    ``advance(duration)`` moves wall-clock time; playback speed is changed
    partway through the run so both speed directions are exercised.
    """
    animator.update(1.63)  # walk already crossed one foot-down this play
    animator.request_transition("run", 0.5, sync_marker="foot_down")
    advance(0.12)  # run blend active (0.12 of 0.5)
    animator.player.state.speed = 1.5
    advance(0.2)  # blend still active; run crosses one foot_down
    # Third action starts before the run blend finishes; it shares the
    # foot-down marker so the contact phase stays replayable end to end.
    animator.request_transition("stride", 0.2, sync_marker="foot_down")
    animator.player.state.speed = 0.5
    advance(0.3)
    animator.player.state.speed = 1.0
    advance(0.4)
    return animator


class ThreeActionSwitchTests(unittest.TestCase):
    def _events(self, animator, advance):
        events = []

        def wrapped(duration):
            advance(duration)
            events.extend(animator.pop_events())

        _scenario(animator, wrapped)
        return events

    def test_speed_changes_do_not_duplicate_or_drop_foot_events(self):
        library = build_demo_library()

        def big_step(duration):
            big.player.update(duration)

        big = CharacterAnimator(library, "walk", STATES)
        big_events = self._events(big, big_step)

        stepped = CharacterAnimator(library, "walk", STATES)
        frame_events = self._events(
            stepped,
            lambda duration: [stepped.update(0.01)
                              for _ in range(round(duration / 0.01))],
        )

        markers = [event["marker"] for event in big_events]
        self.assertEqual(markers, [event["marker"] for event in frame_events])
        # Every occurrence key is unique: no duplicates from the interrupted
        # blend and no missing foot-downs while speed changed mid-way.
        keys = [
            (event["clip"], event["cycle"])
            for event in big_events
            if event["marker"] == "foot_down"
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 0)
        # Walk produced events, run produced events, cancelled unreached
        # markers were never fabricated.
        clips = {event["clip"] for event in big_events}
        self.assertIn("walk", clips)
        self.assertIn("run", clips)

    def test_three_advancement_styles_match_pose_root_and_order(self):
        library = build_demo_library()

        big_player = CharacterAnimator(library, "walk", STATES)
        big_events = []
        big = _scenario(
            big_player,
            lambda d: [big_player.update(d),
                       big_events.extend(big_player.pop_events())],
        )

        frame = CharacterAnimator(library, "walk", STATES)
        frame_events = []

        def frame_advance(duration):
            for _ in range(round(duration / 0.01)):
                frame.update(0.01)
                frame_events.extend(frame.pop_events())

        _scenario(frame, frame_advance)

        def assert_matches(reference, other):
            self.assertAlmostEqual(
                reference.player.state.local_time,
                other.player.state.local_time,
                places=9,
            )
            for axis in range(2):
                self.assertAlmostEqual(
                    reference.root_position[axis],
                    other.root_position[axis],
                    places=9,
                )
            for a, b in zip(reference.current_pose().joints,
                            other.current_pose().joints):
                self.assertAlmostEqual(a, b, places=9)

        assert_matches(big, frame)
        self.assertEqual(
            [e["marker"] for e in big_events],
            [e["marker"] for e in frame_events],
        )

        # Save in the middle of the run blend (after 0.12 of 0.5 s),
        # restore into a fresh player, then finish the exact same scenario.
        saved = CharacterAnimator(library, "walk", STATES)
        saved.update(1.63)
        saved.request_transition("run", 0.5, sync_marker="foot_down")
        for _ in range(12):
            saved.update(0.01)
            saved.pop_events()
        checkpoint = saved.save_state()
        replayed = CharacterAnimator(library, "walk", STATES)
        replayed.load_state(checkpoint)
        for _ in range(20):  # 0.2 wall s at speed 1.5 remaining
            replayed.update(0.01)
            saved.update(0.01)
            replayed.pop_events()
            saved.pop_events()
        for animator in (replayed, saved):
            animator.request_transition("stride", 0.2, sync_marker="foot_down")
            animator.player.state.speed = 0.5
            for _ in range(30):
                animator.update(0.01)
                animator.pop_events()
            animator.player.state.speed = 1.0
            for _ in range(40):
                animator.update(0.01)
                animator.pop_events()
        assert_matches(saved, replayed)
        # The restored checkpoint itself is the mid-blend contact phase.
        self.assertEqual(checkpoint["player"]["clip_name"], "run")
        self.assertLess(
            checkpoint["player"]["blend_elapsed"],
            checkpoint["player"]["blend_duration"],
        )

    def test_contact_history_survives_save_restore(self):
        library = build_demo_library()
        animator = CharacterAnimator(library, "walk", STATES)
        animator.update(0.63)
        animator.request_transition("run", 0.3, sync_marker="foot_down")
        animator.update(0.12)
        animator.request_transition("stride", 0.2, sync_marker="foot_down")
        data = animator.save_state()

        restored = CharacterAnimator(library, "walk", STATES)
        restored.load_state(data)
        self.assertEqual(
            restored.transition_history(), animator.transition_history()
        )


if __name__ == "__main__":
    unittest.main()

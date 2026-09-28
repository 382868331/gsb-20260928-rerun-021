"""Determinism and continuity tests for the animation engine."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anim_engine import AnimInstance, build_library
from anim_engine.rules import TransitionRule, evaluate_rules

DT = 0.01


def pose_distance(p1, p2):
    total = 0.0
    for joint in p1:
        for a, b in zip(p1[joint], p2[joint]):
            total += (a - b) ** 2
    return math.sqrt(total)


def run_steps(instance, n):
    out = []
    for _ in range(n):
        pose = instance.update(DT)
        out.append((dict(pose), tuple(instance.position)))
    return out


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.lib = build_library()

    def test_save_restore_matches_continuous(self):
        ref = AnimInstance(self.lib, "idle")
        ref.transition_to("walk", 0.3)
        reference = run_steps(ref, 40)

        a = AnimInstance(self.lib, "idle")
        a.transition_to("walk", 0.3)
        first = run_steps(a, 20)
        snapshot = a.get_state()
        resumed = AnimInstance(self.lib)
        resumed.set_state(snapshot)
        second = run_steps(resumed, 20)

        for k in range(20):
            pa, xa = first[k]
            pb, xb = reference[k]
            self.assertAlmostEqual(pose_distance(pa, pb), 0.0, places=12)
            self.assertAlmostEqual(xa[0], xb[0], places=12)
        for k in range(20):
            pa, xa = second[k]
            pb, xb = reference[20 + k]
            self.assertAlmostEqual(pose_distance(pa, pb), 0.0, places=12)
            self.assertAlmostEqual(xa[0], xb[0], places=12)

    def test_mid_transition_switch_has_no_pop(self):
        # Baseline: how much does a plain run clip change per frame on its
        # own? A smooth switch must not introduce a jump beyond that motion.
        run_only = AnimInstance(self.lib, "run")
        baseline_pose = 0.0
        baseline_x = 0.0
        prev = run_only.sample_pose()
        for _ in range(80):
            run_only.update(DT)
            cur = run_only.sample_pose()
            baseline_pose = max(baseline_pose, pose_distance(prev, cur))
            prev = cur
        # walk stride 0.6/0.8=0.75 m/s ; run stride 1.2/0.5=2.4 m/s
        baseline_x = 2.4 * DT

        instance = AnimInstance(self.lib, "idle")
        instance.transition_to("walk", 0.4)
        instance.update(DT)
        instance.update(DT)
        before_pose = instance.sample_pose()
        before_x = instance.position[0]
        # First 0.4 s fade is only 0.02 s in when we switch again to run:
        # existing layers keep their weights, new layer starts at zero.
        instance.transition_to("run", 0.4)
        self.assertAlmostEqual(instance.position[0], before_x, places=12)
        self.assertAlmostEqual(
            pose_distance(before_pose, instance.sample_pose()), 0.0, places=12)
        prev_pose, prev_x = before_pose, before_x
        worst_pose = 0.0
        worst_x = 0.0
        for _ in range(80):
            instance.update(DT)
            pose, x = instance.sample_pose(), instance.position[0]
            worst_pose = max(worst_pose, pose_distance(prev_pose, pose))
            worst_x = max(worst_x, x - prev_x)
            prev_pose, prev_x = pose, x
        # The switch path may be a touch smoother than pure run, but any
        # discontinuity would dwarf these per-frame motion bounds.
        self.assertLessEqual(worst_pose, baseline_pose * 1.25 + 1e-9)
        self.assertLessEqual(worst_x, baseline_x * 1.25 + 1e-9)

    def test_root_motion_continuous_across_loop_seam(self):
        instance = AnimInstance(self.lib, "walk")
        duration = self.lib["walk"].duration
        positions = [0.0]
        seams = 0
        prev_t = instance.layers[0].time
        for _ in range(120):
            instance.update(DT)
            positions.append(instance.position[0])
            now_t = instance.layers[0].time % duration
            if now_t < prev_t:
                seams += 1
            prev_t = now_t
        self.assertGreater(seams, 0)
        steps = [b - a for a, b in zip(positions, positions[1:])]
        self.assertTrue(all(s >= 0.0 for s in steps))
        # 0.6 m per 0.8 s cycle at speed 1.0; 1.2 s elapsed -> 0.9 m.
        self.assertAlmostEqual(positions[-1], 0.9, places=6)

    def test_speed_scales_time_and_distance(self):
        slow = AnimInstance(self.lib, "walk", speed=0.5)
        fast = AnimInstance(self.lib, "walk", speed=2.0)
        slow.update(1.0)
        fast.update(1.0)
        self.assertAlmostEqual(fast.position[0], 4.0 * slow.position[0], places=9)
        self.assertAlmostEqual(slow.layers[0].time, 0.5, places=9)
        self.assertAlmostEqual(fast.layers[0].time, 2.0, places=9)

    def test_shared_clips_independent_heads(self):
        actors = [AnimInstance(self.lib, "walk") for _ in range(4)]
        for i, inst in enumerate(actors):
            for _ in range(i * 10):
                inst.update(DT)
        times = [inst.layers[0].time for inst in actors]
        self.assertEqual(len(set(round(t, 9) for t in times)), 4)
        self.assertIs(actors[0].layers[0].clip, actors[3].layers[0].clip)

    def test_conditional_rules(self):
        instance = AnimInstance(self.lib, "idle")
        rules = [
            TransitionRule("run", lambda c: c.get("speed_input", 0.0) > 3.0,
                           duration=0.2),
            TransitionRule("walk", lambda c: c.get("speed_input", 0.0) > 0.5,
                           duration=0.3),
        ]
        fired = evaluate_rules(instance, rules, {"speed_input": 1.0})
        self.assertEqual(fired.target, "walk")
        self.assertIsNone(
            evaluate_rules(instance, rules, {"speed_input": 1.0}))
        evaluate_rules(instance, rules, {"speed_input": 5.0})
        self.assertEqual(instance.layers[-1].clip.name, "run")

    def test_footstep_events_fire_across_loops(self):
        events = []
        instance = AnimInstance(
            self.lib, "run",
            event_callback=lambda clip, name, w: events.append((clip, name)))
        for _ in range(100):
            instance.update(DT)
        self.assertEqual(len(events), 4)
        self.assertTrue(all(clip == "run" for clip, _ in events))


if __name__ == "__main__":
    unittest.main()

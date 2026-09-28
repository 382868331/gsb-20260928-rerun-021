"""Demo: condition-driven transitions, mid-fade re-switch, save/resume.

Run: .venv/bin/python demo.py
"""

import json

from anim_engine import AnimInstance, build_library
from anim_engine.rules import TransitionRule, evaluate_rules


def leg_summary(pose):
    q = pose["l_leg"]
    return f"l_leg=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})"


def main():
    library = build_library()

    rules = [
        TransitionRule("run", lambda c: c["speed_input"] > 3.0, 0.30),
        TransitionRule("walk", lambda c: c["speed_input"] > 0.5, 0.40),
        TransitionRule("idle", lambda c: True, 0.40),
    ]

    footsteps = []
    actor = AnimInstance(
        library, "idle",
        event_callback=lambda clip, name, w: footsteps.append(name))

    other = AnimInstance(library, "walk")
    other.update(0.37)

    timeline = [
        (0.20, {"speed_input": 0.0}),
        (0.40, {"speed_input": 1.2}),
        (0.55, {"speed_input": 5.0}),
        (2.00, {"speed_input": 0.0}),
    ]
    t = 0.0
    dt = 0.05
    print("time  clip(s)                       root_x    pose")
    while t < 3.0:
        for at, ctx in timeline:
            if abs(t - at) < 1e-9:
                fired = evaluate_rules(actor, rules, ctx)
                if fired:
                    print(f"-- t={t:4.2f}s condition -> {fired.target}")
        actor.update(dt)
        active = ",".join(
            f"{l.clip.name}:{l.weight:.2f}" for l in actor.layers)
        pose = actor.sample_pose()
        print(f"{t:4.2f}  {active:28s} {actor.position[0]:7.3f}m  "
              f"{leg_summary(pose)}")
        t = round(t + dt, 4)

    print()
    print("Footstep events received:", footsteps)

    snapshot = json.dumps(actor.get_state())
    actor.update(0.1)
    actor.update(0.1)
    resumed = AnimInstance(library)
    resumed.set_state(json.loads(snapshot))
    resumed.update(0.1)
    resumed.update(0.1)
    print("continuous x=%.9f  resumed x=%.9f" % (actor.position[0], resumed.position[0]))
    p1, p2 = actor.sample_pose(), resumed.sample_pose()
    diff = max(abs(a - b) for joint in p1 for a, b in zip(p1[joint], p2[joint]))
    print("max pose channel diff after resume: %.3e" % diff)


if __name__ == "__main__":
    main()

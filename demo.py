"""Run the built-in short action scenario and print continuity diagnostics."""

from animation_engine import (
    AnimationClip,
    CharacterAnimator,
    Keyframe,
    Marker,
    State,
    Transition,
    build_demo_library,
)


def pose_delta(a, b):
    return max(abs(x - y) for x, y in zip(a.joints, b.joints))


def main():
    library = build_demo_library()
    animator = CharacterAnimator(
        library,
        "idle",
        states=[
            State("idle", speed=1.0),
            State("walk", speed=1.0),
            State("dodge", "step_dodge", speed=1.0),
        ],
        transitions=[
            Transition("idle", "walk", duration=0.4,
                       condition=lambda ctx: ctx.get("want_walk")),
        ],
    )

    dt = 0.05
    context = {"want_walk": False}
    frames = []
    retransition_jump_pose = None
    retransition_jump_velocity = None
    retransition_jump_root = None
    for i in range(18):
        if i == 2:
            context["want_walk"] = True
        # Interrupt idle->walk before it finishes.
        if i == 6:
            before_pose = animator.current_pose()
            before_root = animator.root_position
            animator.request_transition("dodge", duration=0.18)
            after_pose = animator.current_pose()
            after_root = animator.root_position
            retransition_jump_pose = max(
                abs(a - b) for a, b in zip(before_pose.joints, after_pose.joints)
            )
            retransition_jump_velocity = max(
                abs(a - b)
                for a, b in zip(before_pose.root_velocity, after_pose.root_velocity)
            )
            retransition_jump_root = (
                (after_root[0] - before_root[0]) ** 2
                + (after_root[1] - before_root[1]) ** 2
            ) ** 0.5
        pose = animator.update(dt, context)
        frames.append((animator.current_state, pose, animator.root_position))

    max_pose_jump = 0.0
    max_root_jump = 0.0
    previous = None
    for state, pose, root in frames:
        if previous is not None:
            old_pose, old_root = previous
            max_pose_jump = max(max_pose_jump, pose_delta(old_pose, pose))
            root_jump = (
                (root[0] - old_root[0]) ** 2 +
                (root[1] - old_root[1]) ** 2
            ) ** 0.5
            max_root_jump = max(max_root_jump, root_jump)
        previous = (pose, root)

    saved = animator.save_state()
    independent = CharacterAnimator(
        library,
        "idle",
        states=[
            State("idle"),
            State("walk"),
            State("dodge", "step_dodge"),
        ],
    )
    independent.load_state(saved)
    for _ in range(10):
        animator.update(dt)
        independent.update(dt)

    saved_again = animator.save_state()
    restored_again = independent.save_state()
    match = all(
        abs(a - b) < 1e-12
        for a, b in zip(saved_again["player"]["root_position"],
                        restored_again["player"]["root_position"])
    )

    print(f"max per-frame pose change: {max_pose_jump:.6f}")
    print(f"max per-frame root move: {max_root_jump:.6f}")
    print(f"retransition pose jump: {retransition_jump_pose:.12f}")
    print(f"retransition velocity jump: {retransition_jump_velocity:.12f}")
    print(f"retransition root jump: {retransition_jump_root:.12f}")
    print(f"saved/restored independent sampling matches: {match}")
    print(f"final root: {animator.root_position}")

    # Marker-synchronised switch and one-shot external events.
    synced = CharacterAnimator(
        library,
        "walk",
        states=[State("walk"), State("run"),
                State("stride", "walk"), State("attack")],
    )
    synced.update(0.63)
    synced.request_transition("run", duration=0.5, sync_marker="foot_down")
    synced.update(0.12)
    synced.player.state.speed = 1.5
    synced.update(0.2)  # crosses a run foot_down while still blending
    # Third switch before the run blend completes; the phase carries over.
    synced.request_transition("stride", duration=0.2, sync_marker="foot_down")
    synced.player.state.speed = 1.0
    synced.update(0.3)
    synced.request_transition("attack", duration=0.0)
    synced.update(0.6)  # single big step crosses windup then hit
    events = synced.pop_events()
    print(f"foot events before attack: "
          f"{[e['clip'] for e in events if e['marker'] == 'foot_down']}")
    print(f"attack events: {[e['marker'] for e in events
                             if e['marker'] != 'foot_down']}")
    for index, entry in enumerate(synced.transition_history()):
        before, after = entry["before"], entry["after"]
        if before is None:
            continue
        print(
            f"switch {index}: {entry['source']} -> {entry['destination']} "
            f"offset {before['offset']:.3f} -> {after['offset']:.3f} "
            f"grounded {before['grounded']} -> {after['grounded']}"
        )


def _gait_clip(name, duration, speed):
    return AnimationClip(
        name,
        [
            Keyframe(0.0, (0.0,), (speed, 0.0)),
            Keyframe(duration, (0.0,), (speed, 0.0)),
        ],
        loop=True,
        markers=[Marker(0.0, "foot_down", {"foot": "left"})],
    )


def gait_timing_demo():
    """Different-period sync plus uniform vs per-phase speed schedules."""
    library = {"slow": _gait_clip("slow", 1.2, 1.0),
               "fast": _gait_clip("fast", 0.6, 2.0)}
    states = [State("slow"), State("fast")]

    # 1.2 s cycle at 0.72 s switches into a 0.6 s cycle: the contact phase
    # (60%) lands at 0.36 s, not at the raw-offset wrap of 0.12 s.
    synced = CharacterAnimator(library, "slow", states)
    synced.update(0.72)
    synced.request_transition("fast", 0.2, sync_marker="foot_down")
    print(f"1.2s@0.72 -> 0.6s cycle lands at "
          f"{synced.player.state.local_time:.2f}s (phase-matched 0.36)")

    # Uniform playback vs a stance/swing speed schedule, both targeting a
    # 1.0 s cycle: print every footfall time and the distance covered.
    gait = {"gait": _gait_clip("gait", 1.0, 1.0)}
    for label, schedule in (
        ("uniform", None),
        ("variable", [(0.0, 0.5), (0.5, 2.0)]),
    ):
        runner = CharacterAnimator(gait, "gait", [State("gait")])
        if schedule is not None:
            runner.set_speed_schedule(schedule, target_cycle_duration=1.0)
        wall = 0.0
        footfalls = []
        for _ in range(210):
            runner.update(0.01)
            wall += 0.01
            for event in runner.pop_events():
                footfalls.append((wall, runner.root_position[0]))
        stamps = ", ".join(
            f"{t:.2f}s@{x:.3f}m" for t, x in footfalls[:3]
        )
        print(f"{label:8s} footfalls: {stamps} "
              f"(first-cycle distance {footfalls[0][1]:.3f}m)")

    # Changing the schedule mid-blend retimes future footfalls only; a
    # save/restore in the middle continues the same phase progress.
    live = CharacterAnimator(gait, "gait", [State("gait")])
    live.set_speed_schedule([(0.0, 0.5), (0.5, 2.0)],
                            target_cycle_duration=1.0)
    live.update(0.4)
    checkpoint = live.save_state()
    live.set_speed_schedule([(0.0, 2.0)], target_cycle_duration=0.5)
    restored = CharacterAnimator(gait, "gait", [State("gait")])
    restored.load_state(checkpoint)
    restored.set_speed_schedule([(0.0, 2.0)], target_cycle_duration=0.5)
    for _ in range(50):
        live.update(0.01)
        restored.update(0.01)
    print(f"mid-blend reschedule: live x={live.root_position[0]:.6f} "
          f"restored x={restored.root_position[0]:.6f}")


if __name__ == "__main__":
    main()
    gait_timing_demo()

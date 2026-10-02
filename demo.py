"""Run the built-in short action scenario and print continuity diagnostics."""

from animation_engine import (
    CharacterAnimator,
    FootSpec,
    GaitTracker,
    GroundSegment,
    State,
    Terrain,
    Transition,
    build_demo_library,
)
from animation_engine.clip import AnimationClip, Keyframe, Marker
from animation_engine.player import AnimationPlayer


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


def gait_phase_sync_demo():
    """Different cycle lengths: the stride phase, not the offset, carries."""
    long_clip = AnimationClip(
        "long_stride",
        [Keyframe(0.0, (0.0,), (1.0, 0.0)), Keyframe(1.2, (1.0,), (1.0, 0.0))],
        loop=True,
        markers=[Marker(0.0, "foot_down")],
    )
    short_clip = AnimationClip(
        "short_stride",
        [Keyframe(0.0, (0.0,), (1.0, 0.0)), Keyframe(0.6, (1.0,), (1.0, 0.0))],
        loop=True,
        markers=[Marker(0.0, "foot_down")],
    )
    clips = {"long_stride": long_clip, "short_stride": short_clip}
    animator = CharacterAnimator(
        clips, "long_stride", [State("long_stride"), State("short_stride")]
    )
    animator.update(0.72)
    animator.request_transition("short_stride", 0.2, sync_marker="foot_down")
    print("gait phase sync (1.2s cycle @0.72 -> 0.6s cycle): "
          f"lands at {animator.player.state.local_time:.3f}s (expect 0.360)")


def speed_schedule_demo():
    """Uniform vs variable playback: foot-down times and travel distance."""
    library = build_demo_library()

    uniform = CharacterAnimator(library, "walk", [State("walk")])
    # Frame-accurate event clock: poll after each small step.
    clock = {"t": 0.0}
    foot_times, foot_distances = [], []
    last_root = uniform.root_position
    for _ in range(300):
        uniform.update(0.01)
        clock["t"] += 0.01
        for event in uniform.pop_events():
            root = uniform.root_position
            foot_times.append(clock["t"])
            foot_distances.append(
                ((root[0] - last_root[0]) ** 2
                 + (root[1] - last_root[1]) ** 2) ** 0.5
            )
            last_root = root
    print("uniform walk (1.0s cycle):")
    for i, (t, d) in enumerate(zip(foot_times, foot_distances)):
        print(f"  foot_down {i + 1}: t={t:.2f}s  distance={d:.3f}m")

    warped = CharacterAnimator(library, "walk", [State("walk")])
    # Slow the stance half, hurry the swing half, 0.8s target cycle.
    warped.set_speed_schedule([(0.5, 0.5), (1.0, 2.0)], target_duration=0.8)
    clock["t"] = 0.0
    foot_times, foot_distances = [], []
    last_root = warped.root_position
    for _ in range(300):
        warped.update(0.01)
        clock["t"] += 0.01
        for event in warped.pop_events():
            root = warped.root_position
            foot_times.append(clock["t"])
            foot_distances.append(
                ((root[0] - last_root[0]) ** 2
                 + (root[1] - last_root[1]) ** 2) ** 0.5
            )
            last_root = root
    print("variable walk (stance 0.5x, swing 2.0x, 0.8s target cycle):")
    for i, (t, d) in enumerate(zip(foot_times, foot_distances)):
        print(f"  foot_down {i + 1}: t={t:.2f}s  distance={d:.3f}m")

    # Changing the arrangement mid-blend only re-times the future.
    live = CharacterAnimator(
        library, "walk", [State("walk"), State("run")]
    )
    live.update(0.63)
    live.request_transition("run", 0.5, sync_marker="foot_down")
    live.update(0.37)
    before = [e["cycle"] for e in live.pop_events()]
    live.set_speed_schedule([(0.5, 0.25), (1.0, 4.0)])
    live.update(2.125)
    after = [e["cycle"] for e in live.pop_events()]
    print("mid-blend schedule change: fired cycles "
          f"{before} kept, new arrivals {after} follow the new plan")

    # Save/restore mid-blend with a schedule keeps the phase progress.
    checkpoint = live.save_state()
    restored = CharacterAnimator(
        library, "walk", [State("walk"), State("run")]
    )
    restored.load_state(checkpoint)
    for _ in range(10):
        live.update(0.05)
        restored.update(0.05)
    drift = abs(
        live.player.state.local_time - restored.player.state.local_time
    )
    print(f"save/restore with schedule: phase drift {drift:.2e}")


if __name__ == "__main__":
    main()
    gait_phase_sync_demo()
    speed_schedule_demo()


def footstep_demo():
    library = build_demo_library()
    player = AnimationPlayer(library, "walk")
    terrain = Terrain([
        GroundSegment(-1.0, 2.0, 0.0, "floor"),
        # Gap between 2.0 and 2.6: no ground to plant on.
        GroundSegment(2.6, 10.0, 0.4, "step"),
    ])
    tracker = GaitTracker(
        player,
        terrain,
        [
            FootSpec("left", (0.0, 0.5), offset=(0.3, 0.1)),
            FootSpec("right", (0.5, 1.0), offset=(0.3, -0.1)),
        ],
    )
    for _ in range(60):
        tracker.update(0.05)
    print("footsteps on uneven ground (gap 2.0-2.6, step at +0.4m):")
    for record in tracker.contact_records():
        if record["satisfied"]:
            x, h = record["position"]
            print(f"  {record['foot']:5s} planted x={x:.2f} h={h:.1f} "
                  f"slide<={record['max_slide']:.3f}m")
        else:
            print(f"  {record['foot']:5s} UNSATISFIED "
                  f"({record['reason']}) at t={record['plant_time']:.2f}s")
    plans = tracker.evaluate_plans("left")
    for plan in plans:
        print(f"  plan {plan['strategy']}: satisfied={plan['satisfied']} "
              f"position={plan['position']} "
              f"adjustment={plan['stride_adjustment']}")


if __name__ == "__main__":
    footstep_demo()

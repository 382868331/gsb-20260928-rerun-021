"""Run the built-in short action scenario and print continuity diagnostics."""

from animation_engine import (
    CharacterAnimator,
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
        states=[State("walk"), State("run"), State("attack")],
    )
    synced.update(0.63)
    synced.request_transition("run", duration=0.2, sync_marker="foot_down")
    synced.update(0.2)
    synced.request_transition("attack", duration=0.0)
    synced.update(0.6)  # single big step crosses windup then hit
    print(f"marker-synced run entry local time: {synced.player.state.local_time:.3f}")
    print(f"attack events: {[e['marker'] for e in synced.pop_events()]}")


if __name__ == "__main__":
    main()

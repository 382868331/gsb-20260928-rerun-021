from dataclasses import dataclass

from .player import AnimationPlayer


@dataclass(frozen=True)
class State:
    name: str
    clip_name: str = None
    speed: float = 1.0

    def __post_init__(self):
        if self.clip_name is None:
            object.__setattr__(self, "clip_name", self.name)


@dataclass(frozen=True)
class Transition:
    source: str
    destination: str
    duration: float
    condition: callable = None
    entry_time: float = 0.0

    def allowed(self, context) -> bool:
        return self.condition is None or bool(self.condition(context))


class CharacterAnimator:
    """Owns one player instance and evaluates named conditional states."""

    def __init__(self, library, initial_state, states=(), transitions=()):
        self.library = library
        self.states = {}
        self.transitions = {}
        for state in states:
            self.add_state(state)
        if initial_state not in self.states:
            raise ValueError(f"unknown initial state {initial_state!r}")
        self.current_state = initial_state
        initial = self.states[initial_state]
        self.player = AnimationPlayer(library, initial.clip_name)
        self.player.state.speed = initial.speed
        for transition in transitions:
            self.add_transition(transition)

    def add_state(self, state: State):
        self.states[state.name] = state
        self.transitions.setdefault(state.name, [])

    def add_transition(self, transition: Transition):
        if transition.source not in self.states:
            raise ValueError(f"unknown transition source {transition.source!r}")
        if transition.destination not in self.states:
            raise ValueError(f"unknown transition destination {transition.destination!r}")
        self.transitions.setdefault(transition.source, []).append(transition)

    def request_transition(self, destination: str, duration: float,
                          entry_time: float = 0.0) -> bool:
        if destination not in self.states:
            return False
        target = self.states[destination]
        self.current_state = destination
        self.player.transition_to(target.clip_name, duration, entry_time)
        self.player.state.speed = target.speed
        return True

    def evaluate_transitions(self, context=None):
        for transition in self.transitions[self.current_state]:
            if transition.allowed(context):
                self.request_transition(
                    transition.destination,
                    transition.duration,
                    transition.entry_time,
                )
                return transition.destination
        return None

    def update(self, dt: float, context=None):
        self.evaluate_transitions(context)
        return self.player.update(dt)

    @property
    def root_position(self):
        return self.player.root_position

    @property
    def root_heading(self):
        return self.player.root_heading

    def current_pose(self):
        return self.player.current_pose()

    def save_state(self) -> dict:
        return {
            "current_state": self.current_state,
            "player": self.player.save_state(),
        }

    def load_state(self, data: dict):
        self.current_state = data["current_state"]
        target = self.states[self.current_state]
        self.player = AnimationPlayer(self.library, target.clip_name)
        self.player.load_state(data["player"])

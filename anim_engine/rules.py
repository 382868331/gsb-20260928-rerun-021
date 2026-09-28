"""Declarative condition-driven transition rules.

A rule maps a predicate evaluated against an arbitrary context dict (speed
input, grounded flag, state flags...) to a target clip. Rules are evaluated
in priority order; the first matching rule whose target differs from the
current dominant clip triggers a smooth cross-fade.
"""


class TransitionRule:
    def __init__(self, target, predicate, duration=0.3, speed=1.0, name=""):
        self.target = target
        self.predicate = predicate
        self.duration = duration
        self.speed = speed
        self.name = name or target


def current_clip(instance):
    """Name of the layer currently fading in (or the sole layer)."""
    for layer in instance.layers:
        if layer.target == 1.0:
            return layer.clip.name
    return instance.layers[0].clip.name if instance.layers else None


def evaluate_rules(instance, rules, context):
    """Apply the first matching rule. Returns the fired rule or None."""
    active = current_clip(instance)
    for rule in rules:
        if rule.predicate(context) and rule.target != active:
            instance.transition_to(rule.target, rule.duration, rule.speed)
            return rule
    return None

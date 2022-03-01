"""Defines feedback mechanisms for reacting to flow fairness decisions."""

from enum import Enum


class ReactionStrategy(Enum):
    """Defines feedback mechanisms for reacting to flow fairness decisions."""

    AIMD = 0
    MIMD = 1


ALL = [ReactionStrategy.AIMD, ReactionStrategy.MIMD]
_STRATEGY_TO_STR = {ReactionStrategy.AIMD: "aimd", ReactionStrategy.MIMD: "mimd"}
_STR_TO_STRATEGY = {string: strat for strat, string in _STRATEGY_TO_STR.items()}


def to_str(strategy):
    """Convert an instance of this enum to a string."""
    if strategy not in _STRATEGY_TO_STR:
        raise KeyError(f"Unknown reaction strategy: {strategy}")
    return _STRATEGY_TO_STR[strategy]


def to_strat(string):
    """Convert a string to an instance of this enum."""
    if string not in _STR_TO_STRATEGY:
        raise KeyError(f"Unknown reaction strategy: {string}")
    return _STR_TO_STRATEGY[string]


def choices():
    """Get the string representations of this enum's choices."""
    return [to_str(strat) for strat in ALL]


def react_up(strategy, current):
    """Increase some current value acording to a ReactionStrategy."""
    if strategy == ReactionStrategy.AIMD:
        new = current + 10
    elif strategy == ReactionStrategy.MIMD:
        new = current * 1.5
    else:
        raise RuntimeError(f"Unknown reaction strategy: {strategy}")
    return new


def react_down(strategy, current):
    """Decrease some current value acording to a ReactionStrategy."""
    if strategy == ReactionStrategy.AIMD:
        new = current - 10
    elif strategy == ReactionStrategy.MIMD:
        new = current / 2
    else:
        raise RuntimeError(f"Unknown reaction strategy: {strategy}")
    return new

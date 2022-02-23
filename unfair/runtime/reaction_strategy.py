"""Defines feedback mechanisms for reacting to flow fairness decisions."""

from enum import Enum


class ReactionStrategy(Enum):
    """Defines feedback mechanisms for reacting to flow fairness decisions."""

    AIMD = 0
    MIMD = 1


ALL = [ReactionStrategy.AIMD, ReactionStrategy.MIMD]
_STRAT_TO_STR = {ReactionStrategy.AIMD: "aimd", ReactionStrategy.MIMD: "mimd"}
_STR_TO_STRAT = {string: strat for strat, string in _STRAT_TO_STR.items()}


def to_str(strat):
    """Convert an instance of this enum to a string."""
    if strat not in _STRAT_TO_STR:
        raise KeyError(f"Unknown reaction strategy: {strat}")
    return _STRAT_TO_STR[strat]


def to_strat(string):
    """Convert a string to an instance of this enum."""
    if string not in _STR_TO_STRAT:
        raise KeyError(f"Unknown reaction strategy: {string}")
    return _STR_TO_STRAT[string]


def choices():
    """Get the string representations of this enum's choices."""
    return [to_str(strat) for strat in ALL]

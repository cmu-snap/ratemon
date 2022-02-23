"""Defines unfairness mitigation strategies."""

from enum import Enum


class MitigationStrategy(Enum):
    """Defines unfairness mitigation strategies."""

    RWND_TUNING = 0
    ACK_PACING = 1


ALL = [MitigationStrategy.RWND_TUNING, MitigationStrategy.ACK_PACING]
_STRAT_TO_STR = {
    MitigationStrategy.RWND_TUNING: "rwnd",
    MitigationStrategy.ACK_PACING: "pace",
}
_STR_TO_STRAT = {string: strat for strat, string in _STRAT_TO_STR.items()}


def to_str(strat):
    """Convert an instance of this enum to a string."""
    if strat not in _STRAT_TO_STR:
        raise KeyError(f"Unknown mitigation strategy: {strat}")
    return _STRAT_TO_STR[strat]


def to_strat(string):
    """Convert a string to an instance of this enum."""
    if string not in _STR_TO_STRAT:
        raise KeyError(f"Unknown mitigation strategy: {string}")
    return _STR_TO_STRAT[string]


def choices():
    """Get the string representations of this enum's choices."""
    return [to_str(strat) for strat in ALL]

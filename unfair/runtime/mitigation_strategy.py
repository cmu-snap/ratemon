"""Defines unfairness mitigation strategies."""

from enum import IntEnum


class MitigationStrategy(IntEnum):
    """Defines unfairness mitigation strategies."""

    RWND_TUNING = 0
    ACK_PACING = 1


ALL = [MitigationStrategy.RWND_TUNING, MitigationStrategy.ACK_PACING]
_STRATEGY_TO_STR = {
    MitigationStrategy.RWND_TUNING: "rwnd",
    MitigationStrategy.ACK_PACING: "pace",
}
_STR_TO_STRATEGY = {string: strat for strat, string in _STRATEGY_TO_STR.items()}


def to_str(strategy):
    """Convert an instance of this enum to a string."""
    if strategy not in _STRATEGY_TO_STR:
        raise KeyError(f"Unknown mitigation strategy: {strategy}")
    return _STRATEGY_TO_STR[strategy]


def to_strat(string):
    """Convert a string to an instance of this enum."""
    if string not in _STR_TO_STRATEGY:
        raise KeyError(f"Unknown mitigation strategy: {string}")
    return _STR_TO_STRATEGY[string]


def choices():
    """Get the string representations of this enum's choices."""
    return [to_str(strat) for strat in ALL]

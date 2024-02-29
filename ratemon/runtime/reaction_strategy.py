"""Defines feedback mechanisms for reacting to flow fairness decisions."""

from enum import IntEnum
import time


class ReactionStrategy(IntEnum):
    """Defines feedback mechanisms for reacting to flow fairness decisions."""

    AIMD = 0
    MIMD = 1
    FILE = 2


ALL = [ReactionStrategy.AIMD, ReactionStrategy.MIMD, ReactionStrategy.FILE]
_STRATEGY_TO_STR = {
    ReactionStrategy.AIMD: "aimd",
    ReactionStrategy.MIMD: "mimd",
    ReactionStrategy.FILE: "file",
}
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
        # TODO: Change to 1 MSS per RTT
        new = current + 1e6
    elif strategy == ReactionStrategy.MIMD:
        new = current * 1.3
    elif strategy == ReactionStrategy.FILE:
        raise RuntimeError(f'Reaction strategy "{to_str(strategy)}" cannot react.')
    else:
        raise RuntimeError(f"Unknown reaction strategy: {strategy}")
    return new


def react_down(strategy, current):
    """Decrease some current value acording to a ReactionStrategy."""
    if strategy == ReactionStrategy.AIMD:
        new = current / 2
    elif strategy == ReactionStrategy.MIMD:
        new = current / 1.75
    elif strategy == ReactionStrategy.FILE:
        raise RuntimeError(f'Reaction strategy "{to_str(strategy)}" cannot react.')
    else:
        raise RuntimeError(f"Unknown reaction strategy: {strategy}")
    return new


def parse_pacing_schedule(flp):
    """Parse a pacing schedule file into a list.

    The file must be a CSV file where each line is of the form:
        <start time (seconds)>,<RWND>

    The resulting list contains tuples of the form:
        (<start time (seconds)>, <RWND>)
    and is sorted by start time.
    """
    now = time.time()
    schedule = []
    with open(flp, "r", encoding="utf-8") as fil:
        for line in fil:
            line = line.strip()
            if line[0] == "#":
                continue
            toks = line.split(",")
            try:
                schedule.append((now + float(toks[0]), float(toks[1])))
            except ValueError as exc:
                raise RuntimeError(f"Improperly formed schedule line: {line}") from exc

    assert len(schedule) > 0
    return sorted(schedule, key=lambda p: p[0])


def get_scheduled_pacing(schedule):
    """Extract the scheduled RWND value for the current time.

    The schedule is a list as described in parse_pacing_schedule().
    """
    assert len(schedule) > 0
    now = time.time()
    while now > schedule[0][0] and len(schedule) > 1:
        schedule = schedule[1:]
    return schedule[0][1]

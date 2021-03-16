
import itertools


def make_ewma_metric(metric, alpha):
    """ Format the name of an EWMA metric. """
    return f"{metric}-ewma-alpha{alpha}"


def make_win_metric(metric, win):
    """ Format the name of a windowed metric. """
    return f"{metric}-windowed-minRtt{win}"


def make_smoothed_features():
    """ Return a dtype for all EWMA and windowed metrics. """
    return (
        [(make_ewma_metric(metric, alpha), typ)
         for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)] +
        [(make_win_metric(metric, win), typ)
         for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)])


# These metrics do not change.
REGULAR = [
    ("seq", "uint32"),
    ("arrival time us", "int32"),
    ("min RTT us", "int32"),
    ("flow share percentage", "float64"),
    ("interarrival time us", "int32"),
    ("inverse interarrival time 1/us", "float64"),
    ("packets lost since last packet estimate", "int32"),
    ("drop rate at bottleneck queue", "float64"),
    ("retransmission rate", "float64"),
    ("payload B", "int32"),
    ("wire len B", "int32"),
    ("total so far B", "int32"),
    ("RTT estimate us", "int32"),
    ("active flows", "int32"),
    ("bandwidth fair share b/s", "float64")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    ("interarrival time us", "float64"),
    ("inverse interarrival time 1/us", "float64"),
    ("RTT estimate us", "float64"),
    ("RTT estimate ratio", "float64"),
    ("loss rate estimate", "float64"),
    ("mathis model throughput b/s", "float64")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    ("average interarrival time us", "float64"),
    ("inverse average interarrival time 1/us", "float64"),
    ("average throughput b/s", "float64"),
    ("throughput share", "float64"),
    ("average RTT estimate us", "float64"),
    ("average RTT estimate ratio", "float64"),
    ("loss event rate", "float64"),
    ("1/sqrt loss event rate", "float64"),
    ("loss rate estimate", "float64"),
    ("mathis model throughput b/s", "float64"),
    # -1 no applicable (no loss yet), 0 lower than or equal to fair
    # throughput, 1 higher. This is not a windowed metric itself, but
    # is based on the "mathis model throughput b/s" metric.
    ("mathis model label", "int32")
]
# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 1000 for i in range(1, 11)] + [i / 10 for i in range(1, 11)]
# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(11)]

FEATURES = [
    fet for fet, _ in make_smoothed_features()
    if ("alpha0.002" not in fet and
        "alpha0.004" not in fet and
        "alpha0.006" not in fet and
        "alpha0.008" not in fet and
        "alpha0.01" not in fet and
        "alpha0.2" not in fet and
        "alpha0.4" not in fet and
        "alpha0.6" not in fet and
        "alpha0.7" not in fet and
        "alpha0.8" not in fet and
        "alpha0.9" not in fet and
        "tt512" not in fet and
        "tt1024" not in fet and
        "mathis model" not in fet and
        "inverse" not in fet and
        "average throughput" not in fet)]

# Features to store as extra data for each sample.
MATHIS_MODEL_FET = "mathis model label-ewma-alpha0.01"
RTT_ESTIMATE_FET = "RTT estimate us-ewma-alpha0.01"
ARRIVAL_TIME_FET = "arrival time us"
THR_ESTIMATE_FET = "throughput p/s-ewma-alpha0.007"
ACTIVE_FLOWS_FET = "active flows"
EXTRA_FETS = [
    ARRIVAL_TIME_FET, MATHIS_MODEL_FET, RTT_ESTIMATE_FET, THR_ESTIMATE_FET,
    ACTIVE_FLOWS_FET]

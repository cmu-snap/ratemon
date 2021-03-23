""" Defines features. """


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

SEQ_FET = "seq"
ARRIVAL_TIME_FET = "arrival time us"
MIN_RTT_FET = "min RTT us"
INTERARR_TIME_FET = "interarrival time us"
INV_INTERARR_TIME_FET = "inverse interarrival time b/s"
PACKETS_LOST_FET = "packets lost since last packet estimate"
DROP_RATE_FET = "drop rate at bottleneck queue"
RETRANS_RATE_FET = "retransmission rate"
LOSS_RATE_FET = "loss rate estimate"
LOSS_EVENT_RATE_FET = "loss event rate"
SQRT_LOSS_EVENT_RATE_FET = "1/sqrt loss event rate"
PAYLOAD_FET = "payload B"
WIRELEN_FET = "wire len B"
TOTAL_SO_FAR_FET = "total so far B"
RTT_ESTIMATE_FET = "RTT estimate us"
RTT_RATIO_FET = "RTT estimate ratio us"
ACTIVE_FLOWS_FET = "active flows"
BW_FAIR_SHARE_FET = "bandwidth fair share b/s"
TPUT_FET = "throughput b/s"
TOTAL_TPUT_FET = "total throughput b/s"
TPUT_SHARE_FET = "throughput share"
LABEL_FET = "class"
MATHIS_TPUT_FET = "mathis model throughput b/s"
# 0 lower than or equal to fair throughput, 1 higher. This is not a windowed
# metric itself, but is based on the "mathis model throughput b/s" metric.
MATHIS_LABEL_FET = "mathis model label"

# These metrics do not change.
REGULAR = [
    (SEQ_FET, "uint32"),
    (ARRIVAL_TIME_FET, "int32"),
    (RTT_ESTIMATE_FET, "int32"),
    (MIN_RTT_FET, "int32"),
    (RTT_RATIO_FET, "float64"),
    (INTERARR_TIME_FET, "int32"),
    (INV_INTERARR_TIME_FET, "float64"),
    (PACKETS_LOST_FET, "int32"),
    (DROP_RATE_FET, "float64"),
    (RETRANS_RATE_FET, "float64"),
    (PAYLOAD_FET, "int32"),
    (WIRELEN_FET, "int32"),
    (TOTAL_SO_FAR_FET, "int32"),
    (ACTIVE_FLOWS_FET, "int32"),
    (BW_FAIR_SHARE_FET, "float64")
]
# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    (INTERARR_TIME_FET, "float64"),
    (INV_INTERARR_TIME_FET, "float64"),
    (RTT_ESTIMATE_FET, "float64"),
    (RTT_RATIO_FET, "float64"),
    (LOSS_RATE_FET, "float64"),
    (MATHIS_TPUT_FET, "float64")
]
# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    (INTERARR_TIME_FET, "float64"),
    (INV_INTERARR_TIME_FET, "float64"),
    (TPUT_FET, "float64"),
    (TOTAL_TPUT_FET, "float64"),
    (TPUT_SHARE_FET, "float64"),
    (RTT_ESTIMATE_FET, "float64"),
    (RTT_RATIO_FET, "float64"),
    (LOSS_EVENT_RATE_FET, "float64"),
    (SQRT_LOSS_EVENT_RATE_FET, "float64"),
    (LOSS_RATE_FET, "float64"),
    (MATHIS_TPUT_FET, "float64"),
    (MATHIS_LABEL_FET, "int32")
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

#MATHIS_MODEL_FET = "mathis model label-ewma-alpha0.01"
#RTT_ESTIMATE_FET = "RTT estimate us-ewma-alpha0.01"
#TPUT_ESTIMATE_FET = "throughput p/s-ewma-alpha0.007"

# Features to store as extra data for each sample.
EXTRA_FETS = [
    ARRIVAL_TIME_FET,
    ACTIVE_FLOWS_FET,
    make_ewma_metric(MATHIS_LABEL_FET, 0.01),
    make_ewma_metric(RTT_ESTIMATE_FET, 0.01),
    make_win_metric(TPUT_FET, 64),
    make_win_metric(TPUT_SHARE_FET, 64)]

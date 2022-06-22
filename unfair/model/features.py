"""Defines features."""

import itertools

from unfair.model import defaults


def make_ewma_metric(metric, alpha):
    """Format the name of an EWMA metric."""
    return f"{metric}-ewma-alpha{alpha}"


def make_win_metric(metric, win):
    """Format the name of a windowed metric."""
    # The loss event rate (and therefore the Mathis model throughput as well) are based
    # on current RTT not minRTT.
    suffix = (
        "curRtt"
        if metric in {LOSS_EVENT_RATE_FET, MATHIS_TPUT_LOSS_EVENT_RATE_FET}
        else "minRtt"
    )
    return f"{metric}-windowed-{suffix}{win}"


def parse_ewma_metric(metric):
    """Parse the name and alpha of an EWMA metric."""
    toks = metric.split("-ewma-alpha")
    return toks[0], float(toks[1])


def parse_win_metric(metric):
    """Parse the name window size of a windowed metric."""
    toks = metric.split("-windowed-")
    name = toks[0]
    suffix = (
        "curRtt"
        if name in {LOSS_EVENT_RATE_FET, MATHIS_TPUT_LOSS_EVENT_RATE_FET}
        else "minRtt"
    )
    return name, int(toks[1].split(suffix)[1])


def make_smoothed_features():
    """Return a dtype for all EWMA and windowed metrics."""
    smoothed_features = [
        (make_ewma_metric(metric, alpha), typ)
        for (metric, typ), alpha in itertools.product(EWMAS, ALPHAS)
    ] + [
        (make_win_metric(metric, win), typ)
        for (metric, typ), win in itertools.product(WINDOWED, WINDOWS)
    ]
    # Drop features that use loss event rate with a window of less than 4. Loss even
    # rate is defined for a window of 8, so 4 is already a bit small.
    return [
        fet
        for fet in smoothed_features
        if not (
            "windowed" in fet
            and (
                fet.startswith(LOSS_EVENT_RATE_FET)
                or fet.startswith(MATHIS_TPUT_LOSS_EVENT_RATE_FET)
            )
            and parse_win_metric(fet)[1] < 4
        )
    ]


def is_unknowable(metric):
    """Return whether a metric is unknowable by a receiver."""
    # Support both "metric" and ("metric", "type").
    if isinstance(metric, tuple):
        metric = metric[0]
    for fet in UNKNOWABLE_FETS:
        if metric.startswith(fet):
            return True
    return False


def is_knowable(metric):
    """Return whether a metric is knowable by a receiver."""
    return not is_unknowable(metric)


SEQ_FET = "seq"
ARRIVAL_TIME_FET = "arrival time us"
MIN_RTT_FET = "min RTT us"
SRTT_FET = "srtt us"
INTERARR_TIME_FET = "interarrival time us"
INV_INTERARR_TIME_FET = "inverse interarrival time b/s"
PACKETS_LOST_FET = "packets lost since last packet"
PACKETS_LOST_TOTAL_FET = "packets lost total"
DROP_RATE_FET = "drop rate at bottleneck queue"
RETRANS_RATE_FET = "retransmission rate"
LOSS_RATE_FET = "loss rate"
SQRT_LOSS_RATE_FET = "1/sqrt loss rate"
LOSS_EVENT_RATE_FET = "loss event rate"
SQRT_LOSS_EVENT_RATE_FET = "1/sqrt loss event rate"
PAYLOAD_FET = "payload B"
WIRELEN_FET = "wire len B"
TOTAL_SO_FAR_FET = "total so far B"
PAYLOAD_SO_FAR_FET = "payload so far B"
RTT_FET = "RTT us"
RTT_RATIO_FET = "RTT ratio us"
ACTIVE_FLOWS_FET = "active flows"
BW_FAIR_SHARE_FRAC_FET = "bandwidth fair share frac"
BW_FAIR_SHARE_BPS_FET = "bandwidth fair share b/s"
TPUT_FET = "throughput b/s"
TPUT_SHARE_FRAC_FET = "throughput share"
TOTAL_TPUT_FET = "total throughput b/s"
TPUT_FAIR_SHARE_BPS_FET = "throughput fair share b/s"
TPUT_TO_FAIR_SHARE_RATIO_FET = "throughput to fair share ratio"
LABEL_FET = "class"
MATHIS_TPUT_LOSS_RATE_FET = "mathis model throughput b/s; loss rate"
MATHIS_TPUT_LOSS_EVENT_RATE_FET = "mathis model throughput b/s; loss event rate"
# JAINS_FAIRNESS_FET = "Jain's fairness index"

# Additional features used when parking packets.
TS_1_FET = "timestamp 1 us"
TS_2_FET = "timestamp 2 us"

# These metrics do not change.
REGULAR = [
    # Use int64 even though sequence numbers are actually uint32 because we need
    # to be able to support our special value for unknown values, which is
    # negative (-1). However, int32 is too small, because sequence numbers can
    # be as large as 2**32 - 1. Therefore, we use int64.
    (SEQ_FET, "int64"),
    (ARRIVAL_TIME_FET, "int64"),
    (RTT_FET, "int32"),
    (MIN_RTT_FET, "int32"),
    (RTT_RATIO_FET, "float64"),
    (INTERARR_TIME_FET, "int32"),
    (INV_INTERARR_TIME_FET, "float64"),
    (PACKETS_LOST_FET, "int32"),
    (PACKETS_LOST_TOTAL_FET, "int32"),
    (LOSS_RATE_FET, "float64"),
    (SQRT_LOSS_RATE_FET, "float64"),
    # (DROP_RATE_FET, "float64"),
    # (RETRANS_RATE_FET, "float64"),
    (PAYLOAD_FET, "int32"),
    (WIRELEN_FET, "int32"),
    (TOTAL_SO_FAR_FET, "int64"),
    (PAYLOAD_SO_FAR_FET, "int64"),
    (ACTIVE_FLOWS_FET, "int32"),
    (BW_FAIR_SHARE_FRAC_FET, "float64"),
    (BW_FAIR_SHARE_BPS_FET, "float64"),
    (MATHIS_TPUT_LOSS_RATE_FET, "float64"),
    # (JAINS_FAIRNESS_FET, "float64")
]

# These metrics are exponentially-weighted moving averages (EWMAs),
# that are recorded for various values of alpha.
EWMAS = [
    (INTERARR_TIME_FET, "float64"),
    (INV_INTERARR_TIME_FET, "float64"),
    (RTT_FET, "float64"),
    (RTT_RATIO_FET, "float64"),
    (LOSS_RATE_FET, "float64"),
    (SQRT_LOSS_RATE_FET, "float64"),
    (MATHIS_TPUT_LOSS_RATE_FET, "float64"),
]

# These metrics are calculated over an window of packets, for varies
# window sizes.
WINDOWED = [
    (INTERARR_TIME_FET, "float64"),
    (INV_INTERARR_TIME_FET, "float64"),
    (TPUT_FET, "float64"),
    # NOTE: Disabled because not used.
    #
    # (TPUT_SHARE_FRAC_FET, "float64"),
    # (TOTAL_TPUT_FET, "float64"),
    # (TPUT_FAIR_SHARE_BPS_FET, "float64"),
    (TPUT_TO_FAIR_SHARE_RATIO_FET, "float64"),
    (RTT_FET, "float64"),
    (RTT_RATIO_FET, "float64"),
    (LOSS_RATE_FET, "float64"),
    (SQRT_LOSS_RATE_FET, "float64"),
    (LOSS_EVENT_RATE_FET, "float64"),
    (SQRT_LOSS_EVENT_RATE_FET, "float64"),
    (MATHIS_TPUT_LOSS_RATE_FET, "float64"),
    (MATHIS_TPUT_LOSS_EVENT_RATE_FET, "float64"),
]

# The alpha values at which to evaluate the EWMA metrics.
ALPHAS = [i / 1000 for i in range(1, 11)] + [i / 10 for i in range(1, 11)]

# The window durations (multiples of the minimum RTT) at which to
# evaluate the window-based metrics.
WINDOWS = [2**i for i in range(11)]

# These features cannot be calculated by an isolated receiver and therefore
# should not be used as training inputs (i.e., should never be in "in_spc").
UNKNOWABLE_FETS = [
    DROP_RATE_FET,
    RETRANS_RATE_FET,
    ACTIVE_FLOWS_FET,
    BW_FAIR_SHARE_FRAC_FET,
    BW_FAIR_SHARE_BPS_FET,
    TPUT_SHARE_FRAC_FET,
    TOTAL_TPUT_FET,
    TPUT_FAIR_SHARE_BPS_FET,
    TPUT_TO_FAIR_SHARE_RATIO_FET,
    LABEL_FET,
]

# Every possible features.
ALL_FEATURES = REGULAR + make_smoothed_features()

# All features that an isolated receiver may use.
ALL_KNOWABLE_FEATURES = [fet for fet in ALL_FEATURES if is_knowable(fet)]

# The feature to use as the ground truth.
OUT_FET = make_win_metric(TPUT_TO_FAIR_SHARE_RATIO_FET, defaults.CHOSEN_WIN)

# Features to store as extra data for each sample.
EXTRA_FETS = [
    ARRIVAL_TIME_FET,
    RTT_FET,
    ACTIVE_FLOWS_FET,
    make_win_metric(TPUT_FAIR_SHARE_BPS_FET, defaults.CHOSEN_WIN),
    make_win_metric(MATHIS_TPUT_LOSS_EVENT_RATE_FET, defaults.CHOSEN_WIN),
]

# Features used when parsing packets from a PCAP file.
PARSE_PCAP_FETS = [
    (SEQ_FET, "int64"),
    (ARRIVAL_TIME_FET, "int64"),
    (TS_1_FET, "int64"),
    (TS_2_FET, "int64"),
    (PAYLOAD_FET, "int32"),
    (WIRELEN_FET, "int32"),
]

# Features used when parsing received packets.
PARSE_PACKETS_FETS = [
    (SEQ_FET, "int64"),
    (RTT_FET, "int32"),
    (WIRELEN_FET, "int32"),
    (PAYLOAD_FET, "int32"),
    (ARRIVAL_TIME_FET, "int64"),
]


def get_names(fets):
    if len(fets) > 0:
        assert isinstance(
            fets[0], tuple
        ), "Input to get_names() must be feature tuples: (name, type)."
    return [fet[0] for fet in fets]


def feature_names_to_dtype(fet_names):
    return [(fet_name, feature_name_to_type(fet_name)) for fet_name in fet_names]


def feature_name_to_type(target_fet_name):
    for fet_name, fet_type in ALL_FEATURES:
        if target_fet_name == fet_name:
            return fet_type
    raise Exception(f"Unknown feature: {target_fet_name}")


def convert_to_float(dtype):
    return [
        (fet_name, fet_type.replace("int", "float")) for fet_name, fet_type in dtype
    ]


def fill_dependencies(in_spc):
    # If the in_spc contains any Mathis model features, then we need to be sure to add
    # the corresponding loss event rate features.
    fets_to_use = set(in_spc)
    to_add = set()
    for name in fets_to_use:
        if MATHIS_TPUT_LOSS_RATE_FET in name:
            if "ewma" in name:
                new_metric = make_ewma_metric(LOSS_RATE_FET, parse_ewma_metric(name)[1])
            else:
                new_metric = make_win_metric(LOSS_RATE_FET, parse_win_metric(name)[1])
            to_add.add(new_metric)
        if MATHIS_TPUT_LOSS_EVENT_RATE_FET in name:
            if "ewma" in name:
                # This should not happen.
                continue
            else:
                new_metric = make_win_metric(
                    LOSS_EVENT_RATE_FET, parse_win_metric(name)[1]
                )
            to_add.add(new_metric)
        if INV_INTERARR_TIME_FET in name:
            to_add.add(INV_INTERARR_TIME_FET)
            to_add.add(INTERARR_TIME_FET)
        if INTERARR_TIME_FET in name:
            to_add.add(INTERARR_TIME_FET)
        if RTT_RATIO_FET in name:
            to_add.add(RTT_RATIO_FET)
            to_add.add(RTT_FET)
        if LOSS_RATE_FET in name:
            to_add.add(LOSS_RATE_FET)
            to_add.add(PACKETS_LOST_FET)
        if SQRT_LOSS_RATE_FET in name:
            if "ewma" in name:
                to_add.add(make_ewma_metric(LOSS_RATE_FET, parse_ewma_metric(name)[1]))
            elif "windowed" in name:
                to_add.add(make_win_metric(LOSS_RATE_FET, parse_win_metric(name)[1]))
            else:
                to_add.add(LOSS_RATE_FET)
        if SQRT_LOSS_EVENT_RATE_FET in name:
            to_add.add(make_win_metric(LOSS_EVENT_RATE_FET, parse_win_metric(name)[1]))
    combined = fets_to_use | to_add
    if combined == fets_to_use:
        return in_spc
    else:
        return fill_dependencies(tuple(sorted(combined)))

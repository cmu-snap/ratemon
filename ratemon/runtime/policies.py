"""Different types of receiver policy, depending on workload."""

from enum import IntEnum
import logging

from ratemon.model import defaults, models, features, utils
from ratemon.runtime import reaction_strategy


class Policy(IntEnum):
    """Different types of receiver policy, depending on workload."""

    NOPOLICY = 0
    SERVICEPOLICY = 1
    FLOWPOLICY = 2
    STATIC_RWND = 3
    SCHEDULED_RWND = 4


POLICIES = [
    Policy.NOPOLICY,
    # Combine all flows from one sender and enforce fairness between senders,
    # regardless of how many flows they use.
    Policy.SERVICEPOLICY,
    Policy.FLOWPOLICY,
    Policy.STATIC_RWND,
    Policy.SCHEDULED_RWND,
]
_POLICY_TO_STR = {
    Policy.NOPOLICY: "nopolicy",
    Policy.SERVICEPOLICY: "servicepolicy",
    Policy.FLOWPOLICY: "flowpolicy",
    Policy.STATIC_RWND: "staticrwnd",
    Policy.SCHEDULED_RWND: "scheduledrwnd",
}
_STR_TO_POLICY = {string: policy for policy, string in _POLICY_TO_STR.items()}


def to_str(policy):
    """Convert an instance of this enum to a string."""
    if policy not in _POLICY_TO_STR:
        raise KeyError(f"Unknown policy: {policy}")
    return _POLICY_TO_STR[policy]


def to_policy(string):
    """Convert a string to an instance of this enum."""
    if string not in _STR_TO_POLICY:
        raise KeyError(f"Unknown policy: {string}")
    return _STR_TO_POLICY[string]


def choices():
    """Get the string representations of this enum's choices."""
    return [to_str(policy) for policy in POLICIES]


def get_model_for_policy(policy, model_file):
    if policy == Policy.NOPOLICY:
        model = models.VoidModel()
    elif policy == Policy.SERVICEPOLICY:
        model = models.ServicePolicyModel()
    elif policy == Policy.FLOWPOLICY:
        assert model_file is not None
        model = models.load_model(model_file)
    elif policy == Policy.STATIC_RWND:
        model = models.VoidModel()
    elif policy == Policy.SCHEDULED_RWND:
        model = models.VoidModel()
    else:
        raise RuntimeError(f"Unknown policy: {to_str(policy)}")
    return model


def make_decision(
    policy,
    flowkeys,
    net,
    min_rtt_us,
    fets,
    label,
    flow_to_decisions,
    strategy,
    schedule,
):
    """Make a rate control mitigation decision.

    Base the decision on the flow's label and existing decision. Use the flow's features
    to calculate any necessary flow metrics, such as the throughput.
    """
    if policy == Policy.NOPOLICY:
        new_decision = (defaults.Decision.NOT_PACED, None, None)
    elif policy == Policy.SERVICEPOLICY:
        new_decision = make_decision_servicepolicy(flowkeys, min_rtt_us, net, fets)
    elif policy == Policy.FLOWPOLICY:
        assert len(flowkeys) == 1
        new_decision = make_decision_flowpolicy(
            flowkeys[0], min_rtt_us, fets, label, flow_to_decisions, strategy
        )
    elif policy == Policy.STATIC_RWND:
        new_decision = make_decision_staticrwnd(schedule)
    elif policy == Policy.SCHEDULED_RWND:
        raise NotImplementedError(f"Policy {to_str(policy)} not implemented!")
    else:
        raise RuntimeError(f"Unknown policy: {to_str(policy)}")
    return new_decision


def make_decision_servicepolicy(flowkeys, min_rtt_us, net, fets):
    """Make a fairness decision for all flows from a sender.

    Take the Mathis fair throughput and divide it equally between the flows.
    """
    # mathis_tput_bps_ler = fets[-1][
    #     features.make_win_metric(
    #         features.MATHIS_TPUT_LOSS_EVENT_RATE_FET, net.win_size
    #     )
    # ]

    ler = fets[-1][features.make_win_metric(features.LOSS_EVENT_RATE_FET, net.win_size)]
    ler = ler / (4 + 5e4 * ler)

    # lr = fets[-1][
    #     features.make_win_metric(
    #         features.LOSS_RATE_FET, net.win_size)
    # ]

    avg_rtt_us = fets[-1][features.make_win_metric(features.RTT_FET, net.win_size)]

    modified_mathis_tput_bps_ler = utils.safe_mathis_tput_bps(
        defaults.MSS_B,
        avg_rtt_us,
        ler,
        # lr,
    )

    # Divide the Mathis fair throughput equally between the flows.
    per_flow_tput_bps = modified_mathis_tput_bps_ler / len(flowkeys)

    return (
        defaults.Decision.PACED,
        per_flow_tput_bps,
        utils.bdp_B(per_flow_tput_bps, min_rtt_us / 1e6),
        # utils.bdp_B(per_flow_tput_bps, avg_rtt_us / 1e6),
    )

    # # Measure the recent loss rate.
    # loss_rate = fets[-1][
    #     features.make_win_metric(
    #         features.LOSS_RATE_FET, net.win_size)
    # ]

    # if label == defaults.Class.ABOVE_TARGET and loss_rate >= 1e-9:
    #     logging.info("Mode 1")
    #     # This sender is sending too fast according to the Mathis model, and we
    #     # know that the bottleneck is fully utilized because there has been loss
    #     # recently. Force all flows to slow down to the Mathis model fair rate.
    #     new_decision = (
    #         defaults.Decision.PACED,
    #         per_flow_tput_bps,
    #         utils.bdp_B(per_flow_tput_bps, min_rtt_us / 1e6),
    #     )
    # elif np.array(
    #     [
    #         flow_to_decisions[flowkey][0] == defaults.Decision.PACED
    #         for flowkey in flowkeys
    #     ]
    # ).any():
    #     logging.info("Mode 2")
    #     # The current measurement is that the sender is not above target, but at
    #     # least one of its flows is already being paced. If the bottlenck is
    #     # not fully utilized, then allow the flows to speed up.
    #     #
    #     # We use the loss rate to determine whether the bottleneck is fully
    #     # utilized. If the loss rate is 0, then the bottleneck is not fully
    #     # utilized. If there is loss, then the bottleneck is fully utilized.

    #     # Look up the average enforced throughput of the flows that are already
    #     # being paced.
    #     avg_enforced_tput_bps = np.average(
    #         [
    #             flow_to_decisions[flowkey][1]
    #             for flowkey in flowkeys
    #             if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
    #         ]
    #     )
    #     logging.info(
    #         "Average enforced throughput: %.2f Mbps", avg_enforced_tput_bps / 1e6
    #     )

    #     if loss_rate < 1e-9:
    #         logging.info("Mode 2.1")
    #         new_tput_bps = reaction_strategy.react_up(
    #             strategy,
    #             # Base the new throughput on the current mathis model fair
    #             # rate. This will prevent the flows from growing quickly.
    #             # But it will allow a bit of probing that will help drive
    #             # the loss rate down faster and lead to quicker growth in
    #             # the Mathis fair rate.
    #             # per_flow_tput_bps,
    #             # Base the new throughput on the previous enforced throughput.
    #             avg_enforced_tput_bps,
    #             # # Base the new throughput on the observed average per-flow
    #             # # throughput.
    #             # # Note: this did not work because a flow that was spuriously
    #             # # aggressive can steal a lot of bandwidth from other flows.
    #             # fets[-1][
    #             #     features.make_win_metric(
    #             #         features.TPUT_FET, net.win_size
    #             #     )
    #             # ]
    #             # / len(flowkeys),
    #             # ^^^ Bw probing: We give it move tput. If it actually achieves
    #             # higher tput, then we give it more.
    #             # But if it doesn't achieve higher tput, then we don't end up growing
    #             # forever.
    #             # With limitless scaling based on the enforce throughput, the
    #             # throughput will eventually
    #             # cause losses and lead to a huge drop, basically like timeout+slow
    #             # start.
    #         )
    #         new_tput_bps = per_flow_tput_bps
    #         new_decision = (
    #             defaults.Decision.PACED,
    #             new_tput_bps,
    #             utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
    #         )
    #     else:
    #         logging.info("Mode 2.2")
    #         new_tput_bps = reaction_strategy.react_down(
    #             strategy,
    #             avg_enforced_tput_bps,
    #         )
    #         new_decision = (
    #             defaults.Decision.PACED,
    #             new_tput_bps,
    #             utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
    #         )
    #         # We know that the bottleneck is fully utilized because the sender
    #         # experienced loss recently. Preserve the existing per-flow decisions.
    #         # new_decision = None

    # else:
    #     logging.info("Mode 3")
    #     # The sender is not above target or it is above target but the link is not
    #     # fully
    #     # utilized, and none of its flows behaved badly in the past, so leave
    #     # it alone.
    #     new_decision = (defaults.Decision.NOT_PACED, None, None)
    # return new_decision


def make_decision_flowpolicy(
    flowkey, min_rtt_us, fets, label, flow_to_decisions, strategy
):
    """Make a fairness decision for a single flow.

    FIXME: Why are the BDP calculations coming out so small? Is the throughput
           just low due to low application demand?
    """
    logging.info("Label for flow %s: %s", flowkey, label)
    tput_bps = utils.safe_tput_bps(fets, 0, len(fets) - 1)
    if label == defaults.Class.ABOVE_TARGET:
        # This flow is sending too fast. Force the sender to slow down.
        new_tput_bps = reaction_strategy.react_down(
            strategy,
            # If the flow was already paced, then based the new paced throughput on
            # the old paced throughput.
            (
                flow_to_decisions[flowkey][1]
                if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                else tput_bps
            ),
        )
        new_decision = (
            defaults.Decision.PACED,
            new_tput_bps,
            utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
        )
    elif flow_to_decisions[flowkey][0] == defaults.Decision.PACED:
        # We are already pacing this flow.
        if label == defaults.Class.BELOW_TARGET:
            # If we are already pacing this flow but we are being too
            # aggressive, then let it send faster.
            new_tput_bps = reaction_strategy.react_up(
                strategy,
                # If the flow was already paced, then base the new paced
                # throughput on the old paced throughput.
                (
                    flow_to_decisions[flowkey][1]
                    if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                    else tput_bps
                ),
            )
            new_decision = (
                defaults.Decision.PACED,
                new_tput_bps,
                utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
            )
        else:
            # If we are already pacing this flow and it is behaving as desired,
            # then all is well. Retain the existing pacing decision.
            new_decision = flow_to_decisions[flowkey]
    else:
        # This flow is not already being paced and is not above rate, so
        # leave it alone.
        new_decision = (defaults.Decision.NOT_PACED, None, None)
    return new_decision


def make_decision_staticrwnd(schedule):
    assert schedule is not None
    return (
        defaults.Decision.PACED,
        None,
        reaction_strategy.get_static_rwnd(schedule),
    )

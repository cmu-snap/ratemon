// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#ifndef __RATEMON_H
#define __RATEMON_H

// Comment out the below line to disable verbose logging.
// #define RM_VERBOSE

#ifdef RM_VERBOSE
#define RM_PRINTF(...) printf(__VA_ARGS__)
// #define RM_PRINTK(...) bpf_printk(__VA_ARGS__)
#else
#define NDEBUG  // Disable assert() calls.
#define RM_PRINTF(...) ;
// #define RM_PRINTK(...)
#endif

// Max number of flows that BPF can track.
#define RM_MAX_FLOWS 8192
// Map pin paths.
#define RM_FLOW_TO_RWND_PIN_PATH "/sys/fs/bpf/flow_to_rwnd"
#define RM_FLOW_TO_WIN_SCALE_PIN_PATH "/sys/fs/bpf/flow_to_win_scale"
#define RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH \
  "/sys/fs/bpf/flow_to_last_data_time_ns"
// Name of struct_ops CCA that flows must use to be woken up.
#define RM_BPF_CUBIC "bpf_cubic"

// Environment variable that specifies the max number of active flows.
#define RM_MAX_ACTIVE_FLOWS_KEY "RM_MAX_ACTIVE_FLOWS"
// Environment variable that specifies how often to perform flow scheduling.
#define RM_EPOCH_US_KEY "RM_EPOCH_US"
// Duration after which an idle flow will be forcibly paused. 0 disables this
// feature.
#define RM_IDLE_TIMEOUT_US_KEY "RM_IDLE_TIMEOUT_US"
// Environment variable that specifies the number of flows to schedule per
// epoch. "Schedule" can mean either activate or pause. Note that setting this
// greater than 1 effectively switches scheduled RWND tuning from a strict
// timeslice mode (where each flow is active for exactly the scheduling epoch)
// to a loose mode (where flows are batched and the entire batch is active for
// at most the oldest flow's epoch, even if other flows in the batch arrived
// more recently).
#define RM_NUM_TO_SCHEDULE_KEY "RM_NUM_TO_SCHEDULE"
// Environment variable that specifies the start range of REMOTE ports to manage
// using scheduled RWND tuning.
#define RM_MONITOR_PORT_START_KEY "RM_MONITOR_PORT_START"
#define RM_MONITOR_PORT_END_KEY "RM_MONITOR_PORT_END"
// Path to cgroup for attaching sockops programs.
#define RM_CGROUP_KEY "RM_CGROUP"

// Key for use in flow-based maps.
struct rm_flow {
  unsigned int local_addr;
  unsigned int remote_addr;
  unsigned short local_port;
  unsigned short remote_port;
};

#endif /* __RATEMON_H */

#pragma once
// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#ifndef __RATEMON_H
#define __RATEMON_H

// Comment out the below line to disable verbose logging.
#define RM_VERBOSE

#ifdef RM_VERBOSE
#define RM_PRINTF(...) printf(__VA_ARGS__)
// #define RM_PRINTK(...) bpf_printk(__VA_ARGS__)
#else
#define NDEBUG // Disable assert() calls.
#define RM_PRINTF(...)
// #define RM_PRINTK(...)
#endif

// Max number of flows that BPF can track.
#define RM_MAX_FLOWS 8192
// Map pin paths.
#define RM_FLOW_TO_RWND_PIN_PATH "/sys/fs/bpf/flow_to_rwnd"
#define RM_FLOW_TO_WIN_SCALE_PIN_PATH "/sys/fs/bpf/flow_to_win_scale"
#define RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH                                     \
  "/sys/fs/bpf/flow_to_last_data_time_ns"
#define RM_FLOW_TO_KEEPALIVE_PIN_PATH "/sys/fs/bpf/flow_to_keepalive"
#define RM_DONE_FLOWS_PIN_PATH "/sys/fs/bpf/done_flows"
// Name of struct_ops CCA that flows must use to be woken up.
#define RM_BPF_CUBIC "bpf_cubic"

// Environment variable that specifies the max number of active flows.
#define RM_MAX_ACTIVE_FLOWS_KEY "RM_MAX_ACTIVE_FLOWS"
// Environment variable specifying scheduling mode, either "time" or "byte".
#define RM_SCHEDILING_MODE_KEY "RM_SCHEDULING_MODE"
// Environment variable for scheduling mode "time" that specifies how long
// to allow a flow to send per epoch.
#define RM_EPOCH_US_KEY "RM_EPOCH_US"
// Environment variable for scheduling mode "byte" that specifies the number of
// bytes to allow a flow to send per epoch.
#define RM_EPOCH_BYTES_KEY "RM_EPOCH_BYTES"
// Environment variable for scheduling mode "byte" that specifies the total size
// of each incast response in bytes. Assumes that response size is uniform
// across incast senders and known in advance. Used to set grant size so that
// the receiver never grants more than necessary, to avoid needed to withdraw
// grants.
#define RM_RESPONSE_SIZE_KEY "RM_RESPONSE_SIZE_BYTES"
// Duration after which an idle flow will be forcibly paused. 0 disables this
// feature.
#define RM_IDLE_TIMEOUT_US_KEY "RM_IDLE_TIMEOUT_US"
// Environment variable that specifies the start range of REMOTE ports to manage
// using scheduled RWND tuning.
#define RM_MONITOR_PORT_START_KEY "RM_MONITOR_PORT_START"
#define RM_MONITOR_PORT_END_KEY "RM_MONITOR_PORT_END"
// Path to cgroup for attaching sockops programs.
#define RM_CGROUP_KEY "RM_CGROUP"

// Key for use in flow-based maps.
struct rm_flow {
  uint32_t local_addr;
  uint32_t remote_addr;
  uint16_t local_port;
  uint16_t remote_port;
};

// Contains grant / RWND information for a flow.
struct rm_grant_info {
  // If set, then ignore the other fields and use this value as the RWND.
  uint32_t override_rwnd_bytes;
  // If set, then look up the ACK seq, use this to set the
  // grant_seq_num_end_bytes, and then reset this to 0.
  uint32_t new_grant_bytes;
  // The sequence number that the grant ends at, used to calculate the RWND.
  uint32_t grant_seq_num_end_bytes;
};

#endif /* __RATEMON_H */

#pragma once
// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#ifndef __RATEMON_H
#define __RATEMON_H

// Comment out the below line to disable verbose logging.
#define RM_VERBOSE

#ifdef RM_VERBOSE
#define RM_PRINTF(...) printf(__VA_ARGS__)
#else
#define NDEBUG // Disable assert() calls.
#define RM_PRINTF(...)
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
// Environment variable that specifies the grant end percent for early grant
// completion. See struct rm_grant_info for more details.
#define RM_GRANT_DONE_PERCENT_KEY "RM_GRANT_DONE_PERCENT"
// Environment variable that specifies how early to consider a grant done (in
// bytes). See struct rm_grant_info for more details.
#define RM_GRANT_END_BUFFER_BYTES_KEY "RM_GRANT_END_BUFFER_BYTES"

#ifndef UINT32_MAX
#define UINT32_MAX 4294967295U
#endif

// Key for use in flow-based maps.
struct rm_flow {
  uint32_t local_addr;
  uint32_t remote_addr;
  uint16_t local_port;
  uint16_t remote_port;
};

// Contains grant / RWND information for a flow.
struct rm_grant_info {
  // Data pending / yet to be granted. Incremented by libratemon_interp on a new
  // burst request, used at tc egress to fine-tune the last grant, to prevent
  // over-granting at the end of a burst. This is allowed to be negative due to
  // extra grants.
  int ungranted_bytes;
  // If not equal to 2^32-1, then use this value as the RWND and ignore the
  // following grant info. This is unsigned because it is tied to the TCP
  // advertised window, which is an unsigned sequence number.
  uint32_t override_rwnd_bytes;
  // The scheduler has assigned a new grant of this amount.
  int new_grant_bytes;
  // The last sequence number in the granted window, regardless of how much
  // pending data the sender has. Used to set the TCP advertised window. This is
  // unsigned because sequence numbers are unsigned. Take care regarding
  // sequence number wrapping.
  uint32_t rwnd_end_seq;
  // The last sequence number in the granted window, reduced if the sender has
  // less data to send than the grant. This is unsigned because sequence numbers
  // are unsigned. Take care regarding sequence number wrapping.
  uint32_t grant_end_seq;
  // Indicates if the flow has recently completed a grant and should not be
  // re-added to the done_flows map. Used to make sure that a flow is added to
  // the done_flows map at most once per grant.
  bool grant_done;
  // When processing a new grant, consider the grant to be done when this
  // percent of the bytes have been ACKed. Must be in the range [0-100]. If this
  // is equal to 100 (i.e., a grant is never done early), then precisely track
  // all extra grant bytes as counting towards the grant end. If this is less
  // than 100, then extra grants will not extend grant_end_seq, so the actual
  // grant done percent will be less than what is configured here.
  int grant_done_percent;
  // Consider a grant done when the ACKed sequence number is within this many
  // bytes of grant_end_seq. Only used if grant_done_percent == 100.
  int grant_end_buffer_bytes;
};

#endif /* __RATEMON_H */

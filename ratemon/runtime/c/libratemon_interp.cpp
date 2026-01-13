#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <boost/asio/deadline_timer.hpp>
#include <boost/asio/detail/impl/epoll_reactor.hpp>
#include <boost/asio/detail/impl/epoll_reactor.ipp>
#include <boost/asio/detail/impl/scheduler.ipp>
#include <boost/asio/detail/impl/service_registry.hpp>
#include <boost/asio/detail/impl/timer_queue_ptime.ipp>
#include <boost/asio/impl/any_io_executor.ipp>
#include <boost/asio/impl/execution_context.hpp>
#include <boost/asio/impl/io_context.hpp>
#include <boost/asio/impl/io_context.ipp>
#include <boost/asio/io_context.hpp>
#include <boost/date_time/posix_time/posix_time_config.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/time.hpp>
#include <boost/operators.hpp>
#include <boost/system/detail/error_code.hpp>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <experimental/random>
#include <linux/bpf.h>
#include <linux/inet_diag.h>
#include <mutex>
#include <shared_mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <queue>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "constant_time_int_queue.h"
#include "ratemon.h"

struct rm_flow_key {
  // trunk-ignore(clang-tidy/misc-non-private-member-variables-in-classes)
  uint32_t local_addr;
  // trunk-ignore(clang-tidy/misc-non-private-member-variables-in-classes)
  uint32_t remote_addr;
  // trunk-ignore(clang-tidy/misc-non-private-member-variables-in-classes)
  uint16_t local_port;
  // trunk-ignore(clang-tidy/misc-non-private-member-variables-in-classes)
  uint16_t remote_port;

  bool operator==(const rm_flow_key &other) const {
    return local_addr == other.local_addr && remote_addr == other.remote_addr &&
           local_port == other.local_port && remote_port == other.remote_port;
  }
};

namespace std {
template <> struct hash<rm_flow_key> {
  std::size_t operator()(const rm_flow_key &key) const {
    return (std::hash<uint32_t>()(key.local_addr) ^
            std::hash<uint32_t>()(key.remote_addr) ^
            std::hash<uint16_t>()(key.local_port) ^
            std::hash<uint16_t>()(key.remote_port));
  }
};
} // namespace std

// Used to signal the scheduler thread to end.
bool run = true;
// Existing signal handler for SIGINT.
struct sigaction oldact;
// FD for the BPF map "flow_to_rwnd".
int flow_to_rwnd_fd = 0;
// FD for the BPF map "flow_to_win_sca" (short for "flow_to_win_scale").
int flow_to_win_scale_fd = 0;
// FD for the BPF map "flow_to_last_da" (short for "flow_to_last_data_time_ns").
int flow_to_last_data_time_fd = 0;
// FD for the BPF map "flow_to_keepali" (short for "flow_to_keepalive").
int flow_to_keepalive_fd = 0;
// BPF ringbuf to poll for flows that have exhausted their grant.
struct ring_buffer *done_flows_rb;
// Runs async timers for scheduling
boost::asio::io_context io;
// Periodically performs scheduling using timer_callback().
boost::asio::deadline_timer timer(io);
// Manages the io_context.
std::optional<std::thread> scheduler_thread;
// Thread to poll the done_flows ringbuffer for byte-based scheduling.
std::optional<std::thread> ringbuf_poll_thread;
// Protects writes and reads to active_fds_queue, paused_fds_queue,
// fd_to_flow, and flow_to_fd. Use unique_lock for writes and shared_lock for
// reads.
std::shared_mutex lock_scheduler;
// FDs for flows that are are currently active.
std::queue<std::pair<int, boost::posix_time::ptime>> active_fds_queue;
// FDs for flows that are currently paused (RWND = 0 B);
ConstantTimeIntQueue paused_fds_queue;
// Maps file descriptor to rm_flow struct.
std::unordered_map<int, struct rm_flow> fd_to_flow;
std::unordered_map<struct rm_flow_key, int> flow_to_fd;
// The next six are scheduled RWND tuning parameters. See ratemon.h for
// parameter documentation.
int max_active_flows = 5;
std::string scheduling_mode = "byte"; // or "time"
std::string new_burst_mode = "normal"; // or "port" or "single"
std::string single_request_policy = "normal"; // or "pregrant" (only applies when new_burst_mode="single")
int epoch_us = 10000;
int epoch_bytes = 65536;
int idle_timeout_us = 0;
int64_t idle_timeout_ns = 0;
// Burst tracking for single_request_pregrant policy
int current_burst_number = 0;
int burst_flows_remaining = 0;
bool pregrant_done = false;
// Ports in this range (inclusive) will be tracked for scheduling.
uint16_t monitor_port_start = 9000;
uint16_t monitor_port_end = 9999;
// Consider a grant done when the ACKed sequence number is within this many
// bytes of the end of the grant.
int grant_end_buffer_bytes = 0;
// If true, all overridden functions will do nothing and just call the original
// implementations. Controlled by the RM_NOOP_MODE environment variable.
bool noop_mode = false;

// Forward declaration so that setup() resolves. Defined for real below.
bool setup();
// Whether setup has been performed.
bool setup_done = setup();

// Used to set entries in BPF maps.
int zero = 0;
boost::posix_time::seconds one_sec = boost::posix_time::seconds(1);
// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length =
    static_cast<socklen_t>(sizeof(placeholder_cc_info));

// Trigger a pure ACK packet to be sent on this FD by calling getsockopt() with
// TCP_CC_INFO. This only works if the flow is using the CCA BPF_CUBIC.
inline void trigger_ack(int fd) {
  RM_PRINTF("INFO: Triggering ACK for flow FD=%d\n", fd);
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO,
             static_cast<void *>(&placeholder_cc_info),
             &placeholder_cc_info_length);
}

// Jitter the provided value by +/- 12.5%. Returns just the jitter.
inline int jitter(int val) {
  return std::experimental::randint(0,
                                    static_cast<int>(std::roundl(val * 0.25))) -
         static_cast<int>(std::roundl(val * 0.125));
}

std::string ipv4_to_string(uint32_t addr) {
  struct in_addr inaddr {};
  inaddr.s_addr = htonl(addr);
  std::array<char, INET_ADDRSTRLEN> buf{};
  if (inet_ntop(AF_INET, &inaddr, buf.data(), INET_ADDRSTRLEN) != nullptr) {
    return std::string(buf.data());
  }
  return std::string("INVALID_IP");
}

// Pause this flow. Return the number of flows that were paused.
inline int pause_flow(int fd, bool trigger_ack_on_pause = true) {
  if (scheduling_mode == "byte") {
    RM_PRINTF("INFO: Cannot pause flow FD=%d in byte-based scheduling mode\n",
              fd);
    return 1;
  }
  // Pausing a flow means setting its RWND to 0 B.
  struct rm_grant_info grant_info {};
  // Need to look up instead of simply overwriting because unacked_bytes must be
  // preserved.
  int err = bpf_map_lookup_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info);
  if (err != 0) {
    // No existing rm_grant_info, so fill in new info.
    RM_PRINTF("ERROR: Could not find existing grant for flow FD=%d\n", fd);
    return 0;
  }
  grant_info.override_rwnd_bytes = 0;
  err = bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info,
                            BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Could not pause flow FD=%d, err=%d (%s)\n", fd, err,
              strerror(-err));
    return 0;
  }
  if (trigger_ack_on_pause) {
    // Trigger an ACK to be sent on this flow.
    RM_PRINTF("INFO: Triggering ACK for paused flow FD=%d\n", fd);
    trigger_ack(fd);
  }
  RM_PRINTF("INFO: Paused flow FD=%d\n", fd);
  return 1;
}

// Find and pause the flow. Return the number of flows that were paused.
int try_find_and_pause(int fd) {
  RM_PRINTF("INFO: Trying to pause flow FD=%d\n", fd);
  // Temporary variable for storing the front of active_fds_queue.
  std::pair<int, boost::posix_time::ptime> active_fr;
  int active_idx = 0;
  // Check all entries in active_fds_queue once, until we find the one we are
  // looking for.
  while (active_idx < static_cast<int>(active_fds_queue.size())) {
    active_fr = active_fds_queue.front();
    RM_PRINTF("INFO: Checking active flow %d at index %d\n", active_fr.first,
              active_idx);
    active_fds_queue.pop();
    ++active_idx;
    if (active_fr.first == fd) {
      // Pause this flow.
      paused_fds_queue.enqueue(active_fr.first);
      return pause_flow(active_fr.first);
    }
    // Add this flow back to active_fds_queue.
    active_fds_queue.push(active_fr);
    // Examine the next flow in active_fds_queue.
  }
  return 0;
}

// Attempt to activate this flow. Return the number of activated flows.
inline int activate_flow(int fd, bool trigger_ack_on_activate = true) {
  int err = 0;
  if (scheduling_mode == "byte") {
    struct rm_grant_info grant_info {};
    // Check if we already have a rm_grant_info for this flow, and if so update
    // it. Note that bpf_map_lookup_elem() copies the element from the map as
    // opposed to giving us a pointer to it, so we need to write back any
    // updates.
    err = bpf_map_lookup_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info);
    if (err != 0) {
      RM_PRINTF("ERROR: Could not find existing grant for flow FD=%d\n", fd);
      return 0;
    }
    if (grant_info.ungranted_bytes <= 0) {
      RM_PRINTF("INFO: Cannot activate flow FD=%d because it has no ungranted "
                "bytes\n",
                fd);
      return 0;
    }
    grant_info.override_rwnd_bytes = 0xFFFFFFFF;
    grant_info.new_grant_bytes += epoch_bytes;
    grant_info.grant_done = false;
    // Write the new grant info into the map.
    err = bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info,
                              BPF_ANY);
    if (err != 0) {
      RM_PRINTF("ERROR: Could not set grant for flow FD=%d, err=%d (%s)\n", fd,
                err, strerror(-err));
      return 0;
    }
    RM_PRINTF("INFO: Activated flow FD=%d with grant of %d bytes\n", fd,
              epoch_bytes);
  } else if (scheduling_mode == "time") {
    // Remove the RWND limit of 0 that has paused the flow.
    err = bpf_map_delete_elem(flow_to_rwnd_fd, &fd_to_flow[fd]);
    if (err != 0) {
      RM_PRINTF("WARNING: Could not delete RWND clamp for flow FD=%d, "
                "err=%d (%s). The flow might not have been clamped.\n",
                fd, err, strerror(-err));
    }
  }
  if (trigger_ack_on_activate) {
    trigger_ack(fd);
  }
  RM_PRINTF("INFO: Activated FD=%d\n", fd);
  return 1;
}

// Try to find and activate one flow. Returns the number of flows that were
// activated.
int try_activate_one() {
  // Loop until we find a paused flow that is valid (not closed).
  int const num_paused = static_cast<int>(paused_fds_queue.size());
  // Temporary variable for storing the front of paused_fds_queue.
  int pause_fr = 0;
  // Current time (absolute).
  boost::posix_time::ptime const now =
      boost::posix_time::microsec_clock::local_time();
  // New epoch time.
  boost::posix_time::ptime const now_plus_epoch =
      now + boost::posix_time::microseconds(epoch_us);
  int dummy = 0;
  for (int j = 0; j < num_paused; ++j) {
    pause_fr = paused_fds_queue.dequeue();
    // If we do not know about this flow (e.g., been closed), then skip it.
    // trunk-ignore(clang-tidy/clang-diagnostic-error)
    if (!fd_to_flow.contains(pause_fr)) {
      RM_PRINTF("INFO: Skipping activating FD=%d, flow closed\n", pause_fr);
      continue;
    }
    // If this flow is not in the flow_to_keepalive map (bpf_map_lookup_elem()
    // returns negative error code when the flow is not found), then it has no
    // pending data and should be skipped.
    int const err = bpf_map_lookup_elem(flow_to_keepalive_fd,
                                        &fd_to_flow[pause_fr], &dummy);
    RM_PRINTF("INFO: Checking flow FD=%d, dummy=%d, err=%d\n", pause_fr, dummy,
              err);
    if (bpf_map_lookup_elem(flow_to_keepalive_fd, &fd_to_flow[pause_fr],
                            &dummy) != 0) {
      RM_PRINTF("INFO: Skipping activating FD=%d, no pending data\n", pause_fr);
      paused_fds_queue.enqueue(pause_fr);
      continue;
    }
    RM_PRINTF("INFO: Trying to activate flow FD=%d\n", pause_fr);
    if (activate_flow(pause_fr) == 0) {
      // Tried but failed to activate this flow, so put it back in the queue.
      paused_fds_queue.enqueue(pause_fr);
      continue;
    }
    // Successful activation, so put it in the active_fds_queue.
    // Randomly jitter the epoch time by +/- 12.5%.
    active_fds_queue.emplace(
        pause_fr,
        now_plus_epoch + boost::posix_time::microseconds(jitter(epoch_us)));
    return 1;
  }
  return 0;
}

// Try to pause this flow and activate one other flow. Returns the number of
// flows that were activated.
int try_pause_one_activate_one(int fd) {
  // Find the flow in active_fds_queue and remove it
  if (try_find_and_pause(fd) == 0) {
    RM_PRINTF("ERROR: Could not find and/or pause flow FD=%d\n", fd);
    return 0;
  }
  // Then find one flow in paused_fds_queue to restart.
  int const num_activated = try_activate_one();
  if (num_activated == 0) {
    RM_PRINTF("ERROR: Could not activate a flow after pausing FD=%d\n", fd);
    return 0;
  }
  return num_activated;
}

// Call this to check if scheduling should take place, and if so, perform it. If
// there are waiting flows and available capacity, then one will be activated.
// Flows are paused and activated in round-robin order. Each flow is allowed to
// be active for at most epoch_us microseconds. Flows that have been idle for
// longer than idle_timeout_ns will be paused.
//
// There must always be a pending timer event, otherwise the timer thread will
// expire. So this function must always set a new timer event, unless it is
// called because the timer was cancelled or the program is supposed to end.
void timer_callback(const boost::system::error_code &error) {
  RM_PRINTF("INFO: In timer_callback\n");

  // 0. Perform validity checks.
  // If an error (such as a cancellation) triggered this callback, then abort
  // immediately. Do not set another timer.
  if (error) {
    RM_PRINTF("ERROR: timer_callback error: %s\n", error.message().c_str());
    return;
  }
  // If the program has been signalled to stop, then exit. Do not set another
  // timer.
  if (!run) {
    RM_PRINTF("INFO: Program signalled to exit\n");
    return;
  }
  // If setup has not been performed yet, then we cannot perform scheduling.
  // Otherwise, revert to slow check mode.
  if (!setup_done) {
    RM_PRINTF("ERROR: Cannot execute timer callback, setup not done\n");
    if (timer.expires_from_now(one_sec) != 0U) {
      RM_PRINTF("ERROR: Timer unexpectedly cancelled (1)\n");
    }
    timer.async_wait(&timer_callback);
    return;
  }
  // Check that relevant parameters have been set. Otherwise, revert to slow
  // check mode.
  if ((max_active_flows == 0U) || (epoch_us == 0U) || (flow_to_rwnd_fd == 0) ||
      (flow_to_win_scale_fd == 0) || (flow_to_last_data_time_fd == 0) ||
      (flow_to_keepalive_fd == 0)) {
    RM_PRINTF(
        "ERROR: cannot continue, invalid max_active_flows=%u, epoch_us=%u, "
        "flow_to_rwnd_fd=%d, flow_to_win_scale_fd=%d, "
        "flow_to_last_data_time_fd=%d, or "
        "flow_to_keepalive_fd=%d\n",
        max_active_flows, epoch_us, flow_to_rwnd_fd, flow_to_win_scale_fd,
        flow_to_last_data_time_fd, flow_to_keepalive_fd);
    if (timer.expires_from_now(one_sec) != 0U) {
      RM_PRINTF("ERROR: Timer unexpectedly cancelled (2)\n");
    }
    timer.async_wait(&timer_callback);
    return;
  }

  // It is now safe to perform scheduling.
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  RM_PRINTF("INFO: Performing scheduling. active=%lu, paused=%lu\n",
            active_fds_queue.size(), paused_fds_queue.size());

  // Temporary variable for storing the front of active_fds_queue.
  std::pair<int, boost::posix_time::ptime> active_fr;
  // Size of active_fds_queue.
  int active_size = 0;
  // Vector of active flows that we plan to pause.
  std::vector<int> to_pause;
  // Current kernel time (since boot).
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int64_t const ktime_now_ns = ts.tv_sec * 1000000000 + ts.tv_nsec;
  // For measuring idle time.
  int64_t last_data_time_ns = 0;
  int64_t idle_ns = 0;
  // Current time (absolute).
  boost::posix_time::ptime const now =
      boost::posix_time::microsec_clock::local_time();
  // New epoch time.
  boost::posix_time::ptime const now_plus_epoch =
      now + boost::posix_time::microseconds(epoch_us);

  if (static_cast<int>(active_fds_queue.size()) > max_active_flows) {
    RM_PRINTF("FATAL ERROR: active_fds_queue.size()=%zu is larger than "
              "max_active_flows=%d\n",
              active_fds_queue.size(), max_active_flows);
    // This should never happen.
    return;
  }

  // Typically, active_fds_queue will be small and paused_fds_queue will be
  // large. Therefore, it is alright for us to iterate through the entire
  // active_fds_queue (multiple times), but we must iterate through as few
  // elements of paused_fds_queue as possible.

  // 1) Perform a status check on all active flows. It is alright to iterate
  // through all of active_fds_queue.
  active_size = static_cast<int>(active_fds_queue.size());
  for (int i = 0; i < active_size; ++i) {
    active_fr = active_fds_queue.front();
    active_fds_queue.pop();
    // 1.1) If this flow has been closed, remove it.
    // trunk-ignore(clang-tidy/clang-diagnostic-error)
    if (!fd_to_flow.contains(active_fr.first)) {
      continue;
    }
    // 1.2) If idle timeout mode is enabled, then check if this flow is
    // past its idle timeout. Skip this check if there are no paused
    // flows (i.e., no flows seeking activation).
    if (idle_timeout_ns > 0 && !paused_fds_queue.empty()) {
      // Look up this flow's last active time.
      if (bpf_map_lookup_elem(flow_to_last_data_time_fd,
                              &fd_to_flow[active_fr.first],
                              &last_data_time_ns) == 0) {
        // If last_data_time_ns is 0, then this flow has not yet been tracked.
        if (last_data_time_ns == 0U) {
          continue;
        }
        if (last_data_time_ns > ktime_now_ns) {
          // This could be fine...perhaps a packet arrived since we captured
          // the current time above?
          RM_PRINTF("WARNING: FD=%d last data time (%lu ns) is in the future "
                    "compared to our current time (%ld ns) by %ld ns. This is "
                    "probably due to a super recent sneaky packet arrival "
                    "since we recorded the current time.\n",
                    active_fr.first, last_data_time_ns, ktime_now_ns,
                    last_data_time_ns - ktime_now_ns);
        }

        idle_ns = ktime_now_ns - last_data_time_ns;
        RM_PRINTF("INFO: FD=%d now: %ld ns, last data time: %lu ns\n",
                  active_fr.first, ktime_now_ns, last_data_time_ns);
        RM_PRINTF("INFO: FD=%d idle has been idle for %lu ns. timeout is %lu "
                  "ns\n",
                  active_fr.first, idle_ns, idle_timeout_ns);
        // If the flow has been idle for longer than the idle timeout, then
        // pause it. We pause the flow *before* activating a replacement
        // flow because it is by definition not sending data, so we do not
        // risk causing a drop in utilization by pausing it immediately.
        if (idle_ns >= idle_timeout_ns) {
          RM_PRINTF("INFO: Pausing FD=%d due to idle timeout\n",
                    active_fr.first);
          // Remove the flow from flow_to_keepalive, signalling that it no
          // longer has pending demand.
          int const err = bpf_map_delete_elem(flow_to_keepalive_fd,
                                              &fd_to_flow[active_fr.first]);
          if (err != 0) {
            RM_PRINTF("ERROR: Could not delete flow FD=%d from keepalive map, "
                      "err=%d (%s)\n",
                      active_fr.first, err, strerror(-err));
          }
          paused_fds_queue.enqueue(active_fr.first);
          pause_flow(active_fr.first);
          continue;
        }
      }
    }
    // 1.3) If we are using time-based scheduling and the flow has been active
    // for longer than its epoch, then plan to pause it.
    if (scheduling_mode == "time" && now > active_fr.second) {
      if (paused_fds_queue.empty()) {
        // If there are no paused flows, then immediately reactivate this flow.
        // Randomly jitter the epoch time by +/- 12.5%.
        active_fds_queue.emplace(
            active_fr.first,
            now_plus_epoch + boost::posix_time::microseconds(jitter(epoch_us)));
        RM_PRINTF("INFO: Reactivated FD=%d\n", active_fr.first);
        continue;
      }
      // Plan to pause this flow.
      to_pause.push_back(active_fr.first);
    }
    // Put the flow back in the active queue. For this to occur, we know the
    // flow is not idle, and it is either not yet at its epoch time or it has
    // passed its epoch time and it will be extracted and paused later.
    active_fds_queue.push(active_fr);
  }

  RM_PRINTF("INFO: Flows in to_pause:\n");
  for (const auto &fd : to_pause) {
    RM_PRINTF("INFO:     %d\n", fd);
  }

  // 2) Activate flows. Now we can calculate how many flows to activate to reach
  // full capacity. This value is the existing free capacity plus the number of
  // flows we intend to pause. The important part here is that we only look at
  // as many entries in paused_fds_queue as needed.
  int const num_to_activate = max_active_flows -
                              static_cast<int>(active_fds_queue.size()) +
                              static_cast<int>(to_pause.size());
  RM_PRINTF("INFO: Attempting to activate %d flows\n", num_to_activate);
  int num_activated = 0;
  for (int i = 0; i < num_to_activate; ++i) {
    num_activated += try_activate_one();
  }
  if (num_activated != num_to_activate) {
    RM_PRINTF(
        "ERROR: Could not activate as many flows as requested, activated=%d, "
        "requested=%d\n",
        num_activated, num_to_activate);
    // If we could not activate all requested flows, then we will have to pause
    // some of the active flows.
  } else {
    RM_PRINTF("INFO: Activated %d flows\n", num_activated);
  }

  // 3) Pause flows. We need to recalculate the number of flows to pause because
  // we may not have been able to activate as many flows as planned. Recall that
  // it is alright to iterate through all of active_fds_queue.
  RM_PRINTF("INFO: active_fds_queue.size()=%zu, paused_fds_queue.size()=%zu, "
            "max_active_flows=%d\n",
            active_fds_queue.size(), paused_fds_queue.size(), max_active_flows);
  int const num_to_pause =
      std::max(0, static_cast<int>(active_fds_queue.size()) - max_active_flows);
#ifdef RM_VERBOSE
  assert(num_to_pause <= static_cast<int>(to_pause.size()));
#endif
  // For each flow that we are supposed to pause, advance through
  // active_fds_queue until we find it.
  RM_PRINTF("INFO: Pausing %d flows, to_pause contains: %zu\n", num_to_pause,
            to_pause.size());
  active_size = static_cast<int>(active_fds_queue.size());
  for (int i = 0; i < num_to_pause; ++i) {
    try_find_and_pause(to_pause[i]);
  }

  // 4) Check invariants.
#ifdef RM_VERBOSE
  // Cannot have more than the max number of active flows.
  assert(static_cast<int>(active_fds_queue.size()) <= max_active_flows);
  // If there are no active flows, then there should also be no paused flows.
  // No, this is not strictly true anymore. If none of the flows have pending
  // data (i.e., none are in flow_to_keepalive), then they will all be paused.
  // assert(!active_fds_queue.empty() || paused_fds_queue.empty());
#endif

  // 5) Calculate when the next timer should expire.
  boost::posix_time::time_duration when;
  if (active_fds_queue.empty()) {
    // If there are no flows, revert to slow check mode.
    RM_PRINTF("INFO: No flows remaining, reverting to slow check mode\n");
    when = one_sec;
  } else {
    auto const next_epoch_us =
        (active_fds_queue.front().second - now).total_microseconds();
    if (scheduling_mode == "byte") {
      if (idle_timeout_ns == 0U) {
        // If we are not using idle timeout mode...
        RM_PRINTF("INFO: In byte-based scheduling mode but no idle timeout, "
                  "falling back to slow check mode\n");
        when = one_sec;
      } else {
        RM_PRINTF("INFO: In byte-based scheduling mode, scheduling timer for "
                  "next idle timeout\n");
        when = boost::posix_time::microsec(idle_timeout_us);
      }
    } else if (idle_timeout_ns == 0U) {
      // If we are not using idle timeout mode...
      RM_PRINTF("INFO: No idle timeout, scheduling timer for next epoch end\n");
      when = boost::posix_time::microsec(next_epoch_us);
    } else if (idle_timeout_us < next_epoch_us) {
      // If we are using idle timeout mode...
      RM_PRINTF("INFO: Scheduling timer for next idle timeout, sooner than "
                "next epoch end\n");
      when = boost::posix_time::microsec(idle_timeout_us);
    } else {
      RM_PRINTF("INFO: epoch_us=%d, idle_timeout_us=%d, next_epoch_us=%ld\n",
                epoch_us, idle_timeout_us, next_epoch_us);
      RM_PRINTF("INFO: Scheduling timer for next epoch end, sooner than idle "
                "timeout\n");
      when = boost::posix_time::microsec(next_epoch_us);
    }
  }

  // 6) Start the next timer.
  if (timer.expires_from_now(when) != 0U) {
    RM_PRINTF("ERROR: Timer unexpectedly cancelled (3)\n");
  }
  timer.async_wait(&timer_callback);
  // lock is automatically released when it goes out of scope
  RM_PRINTF("INFO: Sleeping until next event in %ld us\n",
            when.total_microseconds());
}

void remove_flow_from_all_maps(struct rm_flow const *flow) {
  if (flow_to_rwnd_fd != 0) {
    bpf_map_delete_elem(flow_to_rwnd_fd, flow);
  }
  if (flow_to_win_scale_fd != 0) {
    bpf_map_delete_elem(flow_to_win_scale_fd, flow);
  }
  if (flow_to_last_data_time_fd != 0) {
    bpf_map_delete_elem(flow_to_last_data_time_fd, flow);
  }
  if (flow_to_keepalive_fd != 0) {
    bpf_map_delete_elem(flow_to_keepalive_fd, flow);
  }
}

// This function is designed to be run in a thread. It is responsible for
// managing the async timers that perform scheduling. The timer events are
// executed by this thread, but they can be scheduled by other threads.
void thread_func() {
  RM_PRINTF("INFO: Scheduler thread started\n");
  if (timer.expires_from_now(one_sec) != 0U) {
    RM_PRINTF("ERROR: Timer unexpectedly cancelled (4)\n");
  }

  timer.async_wait(&timer_callback);
  RM_PRINTF("INFO: Scheduler thread initial sleep\n");
  // Execute the configured events, until there are no more events to execute.
  io.run();

  // Clean up all flows.
  {
    std::unique_lock<std::shared_mutex> lock(lock_scheduler);
    for (const auto &pair : fd_to_flow) {
      remove_flow_from_all_maps(&pair.second);
    }
  }
  RM_PRINTF("INFO: Scheduler thread ended\n");

  // Need to manually free the ring buffer.
  if (done_flows_rb != nullptr) {
    ring_buffer__free(done_flows_rb);
  }

  if (run) {
    RM_PRINTF("ERROR: Scheduled thread ended before program was signalled to "
              "stop\n");
  }
  RM_PRINTF("INFO: Scheduler thread ended\n");
}

// Pre-grant for the next burst when only max_active_flows remain working on current burst.
// This is used in single_request_pregrant policy.
// Caller must hold lock_scheduler.
void pregrant_for_next_burst() {
  RM_PRINTF("INFO: Pre-granting for next burst (burst %d -> %d)\n",
            current_burst_number, current_burst_number + 1);

  int pregranted_count = 0;

  // Grant to all flows that have pending ungranted bytes
  // Caller holds lock, so we can safely iterate fd_to_flow
  for (const auto &entry : fd_to_flow) {
    int const fd = entry.first;
    struct rm_flow const &flow_iter = entry.second;
    // Check if flow has ungranted bytes (indicating burst request was received)
    struct rm_grant_info grant_info {};
    int err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow_iter, &grant_info);
    if (err != 0 || grant_info.ungranted_bytes <= 0) {
      continue;  // No pending data for this flow
    }

    // Pre-grant: give an extra large grant without officially activating the flow
    // The flow remains in its current queue (paused or active) but gets a grant
    // ready for when the next burst request arrives

    // Set a large grant for the next burst (byte mode only)
    grant_info.override_rwnd_bytes = 0xFFFFFFFF;
    grant_info.new_grant_bytes += epoch_bytes;
    grant_info.grant_done = false;
    err = bpf_map_update_elem(flow_to_rwnd_fd, &flow_iter, &grant_info, BPF_ANY);
    if (err != 0) {
      RM_PRINTF("ERROR: Pregrant - could not set grant for flow FD=%d, err=%d (%s)\n",
                fd, err, strerror(-err));
      continue;
    }
    RM_PRINTF("INFO: Pregrant - gave grant of %d bytes to flow FD=%d "
              "(without official activation)\n",
              epoch_bytes, fd);

    // Trigger an ACK to notify the sender about the grant
    trigger_ack(fd);

    pregranted_count++;
  }

  pregrant_done = true;
  RM_PRINTF("INFO: Pre-granted %d flows for next burst\n", pregranted_count);
}

int handle_grant_done(void * /*ctx*/, void *data, size_t data_sz) {
  RM_PRINTF("INFO: In handle_grant_done, data_sz=%zu\n", data_sz);
  if (!setup_done) {
    RM_PRINTF("ERROR: Cannot handle grant done, setup not done\n");
    return 0;
  }

  if (scheduling_mode != "byte") {
    return 0;
  }
  if (data_sz != sizeof(struct rm_flow)) {
    RM_PRINTF("ERROR: Invalid data size %zu, expected %zu\n", data_sz,
              sizeof(struct rm_flow));
    return 0;
  }
  const auto *flow = static_cast<const struct rm_flow *>(data);

  // This flow has exhausted its grant. Remove it from the active flows. It
  // should already be paused (in flow_to_rwnd map with value of 0) because its
  // grant will have been decremented as data arrived.

  // Activate a new flow.
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  rm_flow_key const key = {flow->local_addr, flow->remote_addr,
                           flow->local_port, flow->remote_port};
  auto fd = flow_to_fd.find(key);
  if (fd == flow_to_fd.end()) {
    RM_PRINTF("ERROR: Could not find FD for flow %s:%u->%s:%u\n",
              ipv4_to_string(flow->remote_addr).c_str(), flow->remote_port,
              ipv4_to_string(flow->local_addr).c_str(), flow->local_port);
    return 0;
  }
  RM_PRINTF("INFO: Flow FD=%d has exhausted its grant\n", fd->second);

  // Check if this flow has completed its entire burst (ungranted_bytes <= 0)
  // This is used for pregrant policy to track burst progress
  if (single_request_policy == "pregrant" && new_burst_mode == "single") {
    struct rm_grant_info grant_info {};
    int err = bpf_map_lookup_elem(flow_to_rwnd_fd, flow, &grant_info);
    if (err == 0 && grant_info.ungranted_bytes <= 0) {
      // This flow completed its burst
      burst_flows_remaining--;
      RM_PRINTF("INFO: Flow FD=%d completed burst, %d flows remaining\n",
                fd->second, burst_flows_remaining);

      // Check if we should trigger pre-granting
      if (!pregrant_done && burst_flows_remaining <= max_active_flows &&
          burst_flows_remaining > 0) {
        RM_PRINTF("INFO: Only %d flows remaining (<= max_active_flows=%d), "
                  "triggering pre-grant\n",
                  burst_flows_remaining, max_active_flows);
        pregrant_for_next_burst();
      }
    }
  }

  if (try_pause_one_activate_one(fd->second) == 0) {
    RM_PRINTF("ERROR: Could not pause flow FD=%d and activate another flow\n",
              fd->second);
  }
  return 0;
}

// Function to poll the done_flows ringbuffer. Only used in byte-based
// scheduling mode.
void ringbuf_poll_func() {
  RM_PRINTF("INFO: Ringbuf poll thread started\n");
  while (run && done_flows_rb != nullptr) {
    int const ret = ring_buffer__poll(done_flows_rb, 100 /* timeout ms */);
    if (ret < 0) {
      RM_PRINTF("ERROR: ring_buffer__poll returned %d\n", ret);
      break;
    }
  }
  RM_PRINTF("INFO: Ringbuf poll thread ended\n");
}

// Catch SIGINT and trigger the scheduler thread and timer to end.
void sigint_handler(int signum) {
  switch (signum) {
  case SIGINT:
    RM_PRINTF("INFO: Caught SIGINT\n");
    run = false;
    // If this is not the scheduler thread, then join the scheduler thread.
    if (scheduler_thread &&
        std::this_thread::get_id() == scheduler_thread->get_id()) {
      RM_PRINTF("WARNING: Caught SIGINT in the scheduler thread. Should this "
                "have happened?\n");
    } else if (scheduler_thread && scheduler_thread->joinable()) {
      scheduler_thread->join();
      scheduler_thread.reset();
    }
    if (ringbuf_poll_thread &&
        std::this_thread::get_id() == ringbuf_poll_thread->get_id()) {
      RM_PRINTF("WARNING: Caught SIGINT in the ringbuf poll thread. Should "
                "this have happened?\n");
    } else if (ringbuf_poll_thread && ringbuf_poll_thread->joinable()) {
      ringbuf_poll_thread->join();
      ringbuf_poll_thread.reset();
    }
    RM_PRINTF("INFO: Resetting old SIGINT handler\n");
    sigaction(SIGINT, &oldact, nullptr);
    break;
  default:
    RM_PRINTF("ERROR: Caught signal %d\n", signum);
    break;
  }
  RM_PRINTF("INFO: Re-raising signal %d\n", signum);
  raise(signum);
}

// Read an environment variable as an integer.
bool read_env_int(const char *key, volatile int *dest, bool allow_zero = false,
                  bool allow_neg = false) {
  char *val_str = getenv(key);
  if (val_str == nullptr) {
    RM_PRINTF("ERROR: Failed to query environment variable '%s'\n", key);
    return false;
  }
  int const val_int = atoi(val_str);
  if (!allow_zero and val_int == 0) {
    RM_PRINTF("ERROR: Invalid value for '%s'=%d (must be != 0)\n", key,
              val_int);
    return false;
  }
  if (!allow_neg and val_int < 0) {
    RM_PRINTF("ERROR: Invalid value for '%s'=%d (must be > 0)\n", key, val_int);
    return false;
  }
  *dest = val_int;
  return true;
}

// Read an environment variable as a string.
bool read_env_string(const char *key, std::string &dest) {
  char *val_str = getenv(key);
  if (val_str == nullptr) {
    RM_PRINTF("ERROR: Failed to query environment variable '%s'\n", key);
    return false;
  }
  // Check that the string is not empty.
  if (strlen(val_str) == 0) {
    RM_PRINTF("ERROR: Invalid value for '%s'='%s' (must be non-empty)\n", key,
              val_str);
    return false;
  }
  dest = std::string(val_str);
  return true;
}

// Perform setup (only once for all flows in this process), such as reading
// parameters from environment variables and setting up BPF maps.
bool setup() {
  // Check if no-op mode is enabled first
  const char *noop_env = std::getenv("RM_NOOP_MODE");
  if (noop_env != nullptr && std::string(noop_env) == "yes") {
    noop_mode = true;
    RM_PRINTF("WARNING: Running in no-op mode. All functions will delegate to kernel implementations\n");
    // In no-op mode we still do all other setup, but later we do not perform any
    // ratemon operations.
  }

  // Parameter setup.

  RM_PRINTF("INFO: Performing setup\n");
  // Read environment variables with parameters.
  if (!read_env_int(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows)) {
    return false;
  }
  if (!read_env_string(RM_SCHEDILING_MODE_KEY, scheduling_mode)) {
    return false;
  }
  if (scheduling_mode != "time" && scheduling_mode != "byte") {
    RM_PRINTF("ERROR: Invalid value for '%s'='%s' (must be 'time' or 'byte')\n",
              RM_SCHEDILING_MODE_KEY, scheduling_mode.c_str());
    return false;
  }

  if (!read_env_string(RM_NEW_BURST_MODE_KEY, new_burst_mode)) {
    // Default to "normal" if not specified
    new_burst_mode = "normal";
    RM_PRINTF("INFO: RM_NEW_BURST_MODE not set, defaulting to 'normal'\n");
  }
  RM_PRINTF("INFO: new_burst_mode=%s\n", new_burst_mode.c_str());
  if (new_burst_mode != "normal" && new_burst_mode != "port" &&
      new_burst_mode != "single") {
    RM_PRINTF("ERROR: Invalid new_burst_mode=%s, must be 'normal', 'port', or 'single'\n",
              new_burst_mode.c_str());
    return false;
  }

  // Read and validate single_request_policy
  if (!read_env_string(RM_SINGLE_REQUEST_POLICY_KEY, single_request_policy)) {
    // Default to "normal" if not specified
    single_request_policy = "normal";
    RM_PRINTF("INFO: RM_SINGLE_REQUEST_POLICY not set, defaulting to 'normal'\n");
  }
  RM_PRINTF("INFO: single_request_policy=%s\n", single_request_policy.c_str());
  if (single_request_policy != "normal" && single_request_policy != "pregrant") {
    RM_PRINTF("ERROR: Invalid single_request_policy=%s, must be 'normal' or 'pregrant'\n",
              single_request_policy.c_str());
    return false;
  }

  if (!read_env_int(RM_EPOCH_US_KEY, &epoch_us)) {
    return false;
  }
  if (!read_env_int(RM_EPOCH_BYTES_KEY, &epoch_bytes)) {
    return false;
  }
  if (epoch_bytes < 1) {
    RM_PRINTF("ERROR: Invalid value for '%s'=%d (must be > 0)\n",
              RM_EPOCH_BYTES_KEY, epoch_bytes);
    return false;
  }
  if (!read_env_int(RM_IDLE_TIMEOUT_US_KEY, &idle_timeout_us,
                    true /* allow_zero */, false /* allow_neg */)) {
    return false;
  }
  idle_timeout_ns = static_cast<int64_t>(idle_timeout_us) * 1000;
  // trunk-ignore(clang-tidy/misc-const-correctness)
  int monitor_port_start_ = 0;
  if (!read_env_int(RM_MONITOR_PORT_START_KEY, &monitor_port_start_) ||
      monitor_port_start_ >= 65536) {
    return false;
  }
  monitor_port_start = static_cast<uint16_t>(monitor_port_start_);
  // trunk-ignore(clang-tidy/misc-const-correctness)
  int monitor_port_end_ = 0;
  if (!read_env_int(RM_MONITOR_PORT_END_KEY, &monitor_port_end_) ||
      monitor_port_end_ >= 65536) {
    return false;
  }
  monitor_port_end = static_cast<uint16_t>(monitor_port_end_);
  if (!read_env_int(RM_GRANT_END_BUFFER_BYTES_KEY, &grant_end_buffer_bytes,
                    true /* allow_zero */, false /* allow_neg */)) {
    return false;
  }

  // BPF setup.

  // Look up the FD for the flow_to_rwnd map. We do not need the BPF skeleton
  // for this.
  int err = bpf_obj_get(RM_FLOW_TO_RWND_PIN_PATH);
  if (err < 0) {
    RM_PRINTF("ERROR: Failed to get FD for 'flow_to_rwnd' from path: '%s', "
              "err=%d : %s\n",
              RM_FLOW_TO_RWND_PIN_PATH, -err, strerror(-err));
    return false;
  }
  flow_to_rwnd_fd = err;

  // Look up the FD for the flow_to_win_scale map. We do not need the BPF
  // skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_WIN_SCALE_PIN_PATH);
  if (err < 0) {
    RM_PRINTF("ERROR: Failed to get FD for 'flow_to_win_scale' from path: "
              "'%s', err=%d : %s\n",
              RM_FLOW_TO_WIN_SCALE_PIN_PATH, -err, strerror(-err));
    return false;
  }
  flow_to_win_scale_fd = err;

  // Look up the FD for the flow_to_last_data_time_ns map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH);
  if (err < 0) {
    RM_PRINTF("ERROR: Failed to get FD for 'flow_to_last_data_time_ns' from "
              "path: '%s', err=%d : %s\n",
              RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH, -err, strerror(-err));
    return false;
  }
  flow_to_last_data_time_fd = err;

  // Look up the FD for the flow_to_keepalive map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_KEEPALIVE_PIN_PATH);
  if (err < 0) {
    RM_PRINTF("ERROR: Failed to get FD for 'flow_to_keepalive' from path: "
              "'%s', err=%d : %s\n",
              RM_FLOW_TO_KEEPALIVE_PIN_PATH, -err, strerror(-err));
    return false;
  }
  flow_to_keepalive_fd = err;

  if (scheduling_mode == "byte") {
    // Look up the FD for the done_flows ringbuf. We do not need the BPF
    // skeleton for this.
    err = bpf_obj_get(RM_DONE_FLOWS_PIN_PATH);
    if (err < 0) {
      RM_PRINTF("ERROR: Failed to get FD for 'done_flows' from path: '%s', "
                "err=%d : %s\n",
                RM_DONE_FLOWS_PIN_PATH, -err, strerror(-err));
      return false;
    }
    // Use the ringbuf fd to create a new userspace ringbuf instance. Note that
    // this must be freed with ring_buffer__free(). It also must be freed on
    // setup() failure, but we create the ringbuffer last so this is not an
    // issue.
    done_flows_rb = ring_buffer__new(err, handle_grant_done, nullptr, nullptr);
    if (done_flows_rb == nullptr) {
      RM_PRINTF("ERROR: Failed to create ring buffer\n");
      return false;
    }
    ringbuf_poll_thread.emplace(ringbuf_poll_func);
    RM_PRINTF(
        "INFO: Successfully created ring buffer for byte-based scheduling\n");
  }

  // Catch SIGINT to end the program.
  struct sigaction action {};
  action.sa_handler = sigint_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, &oldact);

  // Launch the scheduler thread.
  scheduler_thread.emplace(thread_func);

  RM_PRINTF("INFO: Setup complete! max_active_flows=%u, epoch_us=%u, "
            "idle_timeout_ns=%lu, monitor_port_start=%u, monitor_port_end=%u\n",
            max_active_flows, epoch_us, idle_timeout_ns, monitor_port_start,
            monitor_port_end);
  return true;
}

// Fill in the four-tuple for this socket.
bool get_flow(int fd, struct rm_flow *flow) {
  if (flow == nullptr) {
    RM_PRINTF("ERROR: Flow pointer is null\n");
    return false;
  }
  // Initialize flow in case we return early.
  flow->local_addr = 0;
  flow->remote_addr = 0;
  flow->local_port = 0;
  flow->remote_port = 0;
  // Determine the four-tuple, which we need to track because RWND tuning is
  // applied based on four-tuple.
  struct sockaddr_in local_addr {};
  socklen_t local_addr_len = sizeof(local_addr);
  // Get the local IP and port.
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-reinterpret-cast)
  if (getsockname(fd, reinterpret_cast<struct sockaddr *>(&local_addr),
                  &local_addr_len) == -1) {
    RM_PRINTF("ERROR: Failed to call 'getsockname'\n");
    return false;
  }
  struct sockaddr_in remote_addr {};
  socklen_t remote_addr_len = sizeof(remote_addr);
  // Get the peer's (i.e., the remote) IP and port.
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-reinterpret-cast)
  if (getpeername(fd, reinterpret_cast<struct sockaddr *>(&remote_addr),
                  &remote_addr_len) == -1) {
    RM_PRINTF("ERROR: Failed to call 'getpeername'\n");
    return false;
  }
  // Fill in the four-tuple.
  flow->local_addr = ntohl(local_addr.sin_addr.s_addr);
  flow->remote_addr = ntohl(remote_addr.sin_addr.s_addr);
  flow->local_port = ntohs(local_addr.sin_port);
  flow->remote_port = ntohs(remote_addr.sin_port);
  return true;
}

// Set the CCA for this socket and make sure it was set correctly.
bool set_cca(int fd, const char *cca) {
  if (setsockopt(fd, SOL_TCP, TCP_CONGESTION, cca, strlen(cca)) == -1) {
    RM_PRINTF("ERROR: Failed to 'setsockopt' TCP_CONGESTION --- is CCA '%s' "
              "loaded?\n",
              cca);
    return false;
  }
  std::array<char, 32> retrieved_cca{};
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(fd, SOL_TCP, TCP_CONGESTION, retrieved_cca.data(),
                 &retrieved_cca_len) == -1) {
    RM_PRINTF("ERROR: Failed to 'getsockopt' TCP_CONGESTION\n");
    return false;
  }
  if (strcmp(retrieved_cca.data(), cca) != 0) {
    RM_PRINTF("ERROR: Failed to set CCA to %s! Actual CCA is: %s\n", cca,
              retrieved_cca.data());
    return false;
  }
  return true;
}

// Add a flow to the flow_to_keepalive map. Must hold lock_scheduler before
// calling this function.
bool set_keepalive(int sockfd) {
  auto flow = fd_to_flow.find(sockfd);
  if (flow == fd_to_flow.end()) {
    // We are not tracking this flow, so ignore it.
    RM_PRINTF("INFO: Ignoring 'send' for FD=%d, not in fd_to_flow\n", sockfd);
    return false;
  }
  int one = 1;
  int const err =
      bpf_map_update_elem(flow_to_keepalive_fd, &flow->second, &one, BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Failed to update flow_to_keepalive for FD=%d, "
              "flow_to_keepalive_fd=%d, err=%d (%s)\n",
              sockfd, flow_to_keepalive_fd, err, strerror(-err));
    return false;
  }
  RM_PRINTF("INFO: Updated flow_to_keepalive for FD=%d\n", sockfd);
  return true;
}

// Verify that an addr is IPv4.
int check_family(const struct sockaddr *addr) {
  if (addr != nullptr && addr->sa_family != AF_INET) {
    RM_PRINTF("WARNING: Got non-AF_INET sa_family=%u\n", addr->sa_family);
    if (addr->sa_family == AF_INET6) {
      RM_PRINTF("WARNING: (continued) got AF_INET6\n");
    }
    return -1;
  }
  return 0;
}

// This should happen during the handshake, in either accept() or connect().
void register_fd_for_monitoring(int fd) {
  // Look up the four-tuple.
  struct rm_flow flow {};
  if (!get_flow(fd, &flow)) {
    return;
  }
  RM_PRINTF("INFO: Found flow FD=%d: %s:%u->%s:%u\n", fd,
            ipv4_to_string(flow.remote_addr).c_str(), flow.remote_port,
            ipv4_to_string(flow.local_addr).c_str(), flow.local_port);
  // Ignore flows that are not in the monitor port range.
  if (flow.remote_port < monitor_port_start ||
      flow.remote_port > monitor_port_end) {
    RM_PRINTF(
        "INFO: Ignoring flow on remote port %u, not in monitor port range: "
        "[%u, %u]\n",
        flow.remote_port, monitor_port_start, monitor_port_end);
    return;
  }
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  fd_to_flow[fd] = flow;
  rm_flow_key const key = {flow.local_addr, flow.remote_addr, flow.local_port,
                           flow.remote_port};
  flow_to_fd[key] = fd;
  // Change the CCA to BPF_CUBIC.
  if (!set_cca(fd, RM_BPF_CUBIC)) {
    return;
  }

  // Create an entry in flow_to_last_data_time_ns for this flow so that the
  // kprobe program knows to start tracking this flow.
  int const err = bpf_map_update_elem(flow_to_last_data_time_fd,
                                      &fd_to_flow[fd], &zero, BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Failed to create entry in flow_to_last_data_time_fd for "
              "FD=%d, err=%d (%s)\n",
              fd, err, strerror(-err));
    return;
  }

  // All flows start paused and are added to scheduler queues.
  // In port mode, initial grants are based on port numbers (in handle_send),
  // but flows are still tracked in queues for when normal scheduling takes over
  // after the flash grants complete.
  paused_fds_queue.enqueue(fd);
  pause_flow(fd);
  RM_PRINTF("INFO: Flow FD=%d registered and added to paused queue\n", fd);
  // lock is automatically released when it goes out of scope
}

// A new request is being sent on this connection. Perform scheduling for the
// flow to determine whether the outgoing request will carry an RWND that
// initially activates or pauses the flow. This happens on a new request to
// send(), NOT in the handshake (accept() or connect()). The caller must hold
// lock_scheduler.
//
// This function can be called between calls to the scheduling timer. Race
// conditions are avoided by the caller holding lock_scheduler. This must leave
// all global state in a safe state.
//
// We need to determine whether the flow is already active or paused. If it is
// active, then do nothing. For fairness, do not refresh its timer/grant. If
// there is space to activate it and it is paused, then do so. If there is no
// space to activate it and it is paused, then do nothing.
bool schedule_on_new_request(int fd) {
  if (!paused_fds_queue.contains(fd)) {
    // The flow is not paused, so it must be active already. Do nothing.
    RM_PRINTF("INFO: Flow FD=%d is already active, doing nothing\n", fd);
    return true;
  }

  if (static_cast<int>(active_fds_queue.size()) >= max_active_flows) {
    RM_PRINTF("INFO: Flow FD=%d is paused and there is no space to "
              "activate it, so doing nothing.\n",
              fd);
    return true;
  }

  // The flow is paused but we can activate it.
  RM_PRINTF("INFO: Flow FD=%d is not active, but there is space to activate "
            "it, so activating it now\n",
            fd);
  if (!paused_fds_queue.find_and_delete(fd)) {
    RM_PRINTF("ERROR: Could not find and delete flow FD=%d from paused "
              "queue, this is a bug!\n",
              fd);
    return false;
  }

  // Activate the flow, but do not trigger an ACK because this scheduling was
  // triggered by an outgoing burst request, and the grant will be carried in
  // the burst request itself. In practice, this call does nothing for
  // time-based scheduling, but it is included for completeness and to get a log
  // indicating the flow was activated.
  boost::posix_time::ptime const now =
      boost::posix_time::microsec_clock::local_time();
  active_fds_queue.emplace(
      fd, now + boost::posix_time::microseconds(epoch_us) +
              boost::posix_time::microseconds(jitter(epoch_us)));
  if (activate_flow(fd, false /* trigger_ack_on_activate */) == 0) {
    RM_PRINTF("ERROR: Failed to activate flow FD=%d\n", fd);
    return false;
  }
  RM_PRINTF("INFO: Activated flow FD=%d\n", fd);

  if (active_fds_queue.size() == 1) {
    // There were no flows active, so we need to exit slow check mode.
    if (timer.expires_from_now(active_fds_queue.front().second - now) != 1) {
      // Timers usually expire naturally, so there is never a timer to cancel.
      // This is a special case where we have a new flow and therefore need to
      // cancel the existing timer and reset it with a new duration. We only
      // do this if there is a single active flow, which we just put in
      // active_fds_queue. If there are multiple active flows, then the timer
      // was already correctly tracking the existing head of the queue.
      // Basically, this cancels slow-check mode and starts epoch-based
      // checks.
      RM_PRINTF("ERROR: Should have cancelled 1 timer. If you are seeing "
                "this, it is possible that the first flow to be tracked was "
                "started before the scheduler thread could launch.\n");
      return false;
    }
    // The above call to timer.expires_from_now() will have cancelled any
    // other pending async_wait(), so we are safe to call async_wait() again.
    timer.async_wait(&timer_callback);
    RM_PRINTF("INFO: Exiting slow check mode.\n");
  }
  return true;
}

// Handle send in normal mode - uses existing queue-based scheduling logic.
// Uses lock only for queue operations, not for BPF map accesses.
static bool handle_send_normal_mode(int sockfd, const void *buf, size_t len) {
  // Update the flow_to_keepalive map to indicate that this flow has pending
  // data. BPF map has its own locking.
  if (!set_keepalive(sockfd)) {
    RM_PRINTF("ERROR: Failed to update keepalive for FD=%d\n", sockfd);
    return false;
  }

  // If we are doing byte-based scheduling, then track the size of this request.
  // BPF map operations don't need our lock.
  if (scheduling_mode == "byte" && len == 2 * sizeof(int)) {
    // trunk-ignore(clang-tidy/google-readability-casting)
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
    int *buf_int = (int *)buf;
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-bounds-pointer-arithmetic)
    int const bytes = buf_int[0];
    RM_PRINTF("INFO: FD=%d has %d bytes pending\n", sockfd, bytes);

    struct rm_flow flow;
    {
      std::shared_lock<std::shared_mutex> lock(lock_scheduler);
      auto flow_it = fd_to_flow.find(sockfd);
      if (flow_it == fd_to_flow.end()) {
        RM_PRINTF("ERROR: Flow FD=%d not found in fd_to_flow\n", sockfd);
        return false;
      }
      flow = flow_it->second;
    }

    struct rm_grant_info grant_info {};
    int err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
    if (err != 0) {
      RM_PRINTF(
          "INFO: Could not find existing grant info for flow FD=%d, creating "
          "new grant info\n",
          sockfd);
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes = 0;
      grant_info.rwnd_end_seq = 0;
      grant_info.grant_end_seq = 0;
      grant_info.grant_done = true;
      grant_info.grant_end_buffer_bytes = grant_end_buffer_bytes;
    }
    grant_info.ungranted_bytes += bytes;
    err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
    if (err != 0) {
      RM_PRINTF("ERROR: Could not update ungranted bytes for flow FD=%d, "
                "err=%d (%s)\n",
                sockfd, err, strerror(-err));
      return false;
    }
  }

  // Use existing queue-based scheduling logic (requires lock)
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  if (!schedule_on_new_request(sockfd)) {
    RM_PRINTF("ERROR: Failed to schedule flow for FD=%d\n", sockfd);
    return false;
  }

  return true;
}

// Handle send in single mode - one send() per host represents all flows from that host.
// This mode is for IBG's single_request feature where the receiver sends one burst
// request to each sender host, and that single send() should update state for all
// flows from that host.
// Two policies:
// - "normal": Perform scheduling on every send() call
// - "pregrant": Perform scheduling only on first burst; subsequent bursts use pre-grants
static bool handle_send_single_mode(int sockfd, const void *buf, size_t len) {
  struct rm_flow flow {};
  if (!get_flow(sockfd, &flow)) {
    RM_PRINTF("ERROR: Could not get flow for FD=%d\n", sockfd);
    return false;
  }

  // Get the remote address from the calling flow
  uint32_t const remote_addr = flow.remote_addr;

  // Parse burst_number from the burst request message
  // Format is: [burst_number, bytes, wait_us]
  int burst_number = -1;
  if (len == 3 * sizeof(int)) {
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-reinterpret-cast)
    const int *buf_int = reinterpret_cast<const int *>(buf);
    burst_number = buf_int[0];
  }

  RM_PRINTF("INFO: Single mode - send() called for FD=%d from host %s, "
            "received_burst_number=%d, current_burst_number=%d, policy=%s\n",
            sockfd, ipv4_to_string(remote_addr).c_str(),
            burst_number, current_burst_number, single_request_policy.c_str());

  // Acquire lock to access fd_to_flow and global state
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);

  // Initialize burst tracking on first burst
  if (current_burst_number == 0 && burst_flows_remaining == 0) {
    // Count total flows to initialize burst_flows_remaining
    burst_flows_remaining = fd_to_flow.size();
    pregrant_done = false;
    RM_PRINTF("INFO: First burst - initialized burst_flows_remaining=%d\n",
              burst_flows_remaining);
  }

  // Determine whether to perform scheduling
  bool should_schedule = false;
  if (single_request_policy == "normal") {
    // Normal policy: always schedule
    should_schedule = true;
  } else if (single_request_policy == "pregrant") {
    // Pregrant policy: only schedule on first burst
    should_schedule = (current_burst_number == 0);
  }

  // Find all flows from the same remote host and update their state
  int flows_updated = 0;
  for (auto &entry : fd_to_flow) {
    int const fd = entry.first;
    struct rm_flow const &flow_iter = entry.second;

    // Check if this flow is from the same remote host
    if (flow_iter.remote_addr == remote_addr) {
      flows_updated++;

      RM_PRINTF("INFO: Single mode - updating flow FD=%d (remote_port=%u) "
                "from host %s\n",
                fd, flow_iter.remote_port, ipv4_to_string(remote_addr).c_str());

      // Set keepalive for this flow (BPF map - has internal locking)
      if (!set_keepalive(fd)) {
        RM_PRINTF("ERROR: Could not set keepalive for flow FD=%d\n", fd);
        // Continue with other flows rather than failing completely
      }

      // Update ungranted bytes in BPF map (BPF map - has internal locking)
      int err = 0;
      struct rm_grant_info grant_info {};
      err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow_iter, &grant_info);
      if (err != 0) {
        RM_PRINTF("WARNING: Could not find grant info for flow FD=%d, "
                  "creating new entry\n", fd);
        grant_info = {};
      }

      // Add the burst size to ungranted bytes
      grant_info.ungranted_bytes += static_cast<int>(len);
      err = bpf_map_update_elem(flow_to_rwnd_fd, &flow_iter, &grant_info, BPF_ANY);
      if (err != 0) {
        RM_PRINTF("ERROR: Could not update grant info for flow FD=%d, "
                  "err=%d (%s)\n", fd, err, strerror(-err));
        // Continue with other flows
      }

      RM_PRINTF("INFO: Single mode - flow FD=%d now has %d ungranted bytes\n",
                fd, grant_info.ungranted_bytes);

      // Perform scheduling if policy allows (requires lock held)
      if (should_schedule) {
        // Call schedule_on_new_request for this flow
        // This will activate the flow if there's capacity, otherwise leave it paused
        if (!schedule_on_new_request(fd)) {
          RM_PRINTF("WARNING: schedule_on_new_request failed for flow FD=%d\n", fd);
          // Continue with other flows
        }

        // Trigger an ACK for flows that are activated, except for the calling flow
        // (sockfd) which will carry the grant in the burst request itself.
        if (fd != sockfd) {
          // NOTE: These ACKs will probably go out slowly due to performance issues with
          // sending packets on many flows, so the first burst will have bad performance.
          // This is okay. We care about steady state (later bursts).
          trigger_ack(fd);
        } else {
          RM_PRINTF("INFO: Single mode - skipping ACK trigger for calling flow FD=%d "
                    "(grant will be in burst request)\n", fd);
        }
      } else {
        RM_PRINTF("INFO: Single mode (pregrant) - skipping scheduling for FD=%d "
                  "(will use pre-grant from previous burst)\n", fd);
      }
    }
  }

  // Update burst tracking based on received burst_number
  if (flows_updated > 0 && burst_number >= 0) {
    if (burst_number > current_burst_number) {
      // New burst detected
      current_burst_number = burst_number;
      pregrant_done = false;
      burst_flows_remaining = fd_to_flow.size();
      RM_PRINTF("INFO: Starting burst %d, reset burst_flows_remaining=%d\n",
                current_burst_number, burst_flows_remaining);
    } else if (burst_number < current_burst_number) {
      RM_PRINTF("WARNING: Received burst_number=%d but current_burst_number=%d\n",
                burst_number, current_burst_number);
    }
  }

  RM_PRINTF("INFO: Single mode - updated %d flows from host %s\n",
            flows_updated, ipv4_to_string(remote_addr).c_str());

  if (flows_updated == 0) {
    RM_PRINTF("WARNING: Single mode - no flows found for host %s (only found FD=%d)\n",
              ipv4_to_string(remote_addr).c_str(), sockfd);
  }

  return true;
}

// Handle send in port mode - makes decisions based on port numbers.
// Provides initial "flash" grants, then queue-based scheduling takes over.
// Uses shared lock for reading flow info, then unique lock for queue manipulation.
static bool handle_send_port_mode(int sockfd, const void *buf, size_t len) {
  // First, extract needed information with a shared lock
  uint16_t remote_port;
  struct rm_flow flow;
  {
    std::shared_lock<std::shared_mutex> lock(lock_scheduler);

    auto flow_it = fd_to_flow.find(sockfd);
    if (flow_it == fd_to_flow.end()) {
      RM_PRINTF("ERROR: Flow FD=%d not found in fd_to_flow\n", sockfd);
      return false;
    }
    flow = flow_it->second;
    remote_port = flow.remote_port;
  }
  // Lock released - now we can operate without blocking other threads

  // Update the flow_to_keepalive map (BPF map, has its own synchronization)
  int one = 1;
  int err = bpf_map_update_elem(flow_to_keepalive_fd, &flow, &one, BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Could not update flow_to_keepalive for FD=%d, "
              "err=%d (%s)\n",
              sockfd, err, strerror(-err));
    return false;
  }
  RM_PRINTF("INFO: Updated flow_to_keepalive for FD=%d\n", sockfd);

  // If we are doing byte-based scheduling, then track the size of this request.
  if (scheduling_mode == "byte" && len == 2 * sizeof(int)) {
    // trunk-ignore(clang-tidy/google-readability-casting)
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
    int *buf_int = (int *)buf;
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-bounds-pointer-arithmetic)
    int const bytes = buf_int[0];
    RM_PRINTF("INFO: FD=%d has %d bytes pending\n", sockfd, bytes);

    struct rm_grant_info grant_info {};
    err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
    if (err != 0) {
      RM_PRINTF(
          "INFO: Could not find existing grant info for flow FD=%d, creating "
          "new grant info\n",
          sockfd);
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes = 0;
      grant_info.rwnd_end_seq = 0;
      grant_info.grant_end_seq = 0;
      grant_info.grant_done = true;
      grant_info.grant_end_buffer_bytes = grant_end_buffer_bytes;
    }
    grant_info.ungranted_bytes += bytes;
    err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
    if (err != 0) {
      RM_PRINTF("ERROR: Could not update ungranted bytes for flow FD=%d, "
                "err=%d (%s)\n",
                sockfd, err, strerror(-err));
      return false;
    }
  }

  // Port mode: make scheduling decision based on remote port number.
  // This provides the initial "flash" grant. After this grant completes,
  // handle_grant_done will transition the flow to normal queue-based scheduling.
  // NOTE: We do NOT touch queue data structures here. The queues will show all
  // flows as paused during flash grants, which is fine - normal scheduling takes
  // over when flash grants complete.
  uint16_t const port_range_end = monitor_port_start + max_active_flows;

  if (remote_port >= monitor_port_start && remote_port < port_range_end) {
    // This flow gets a flash grant - set grant in BPF maps only, don't touch queues
    RM_PRINTF("INFO: Port mode - giving flash grant to flow FD=%d with "
              "remote_port=%u (in range [%u, %u))\n",
              sockfd, remote_port, monitor_port_start, port_range_end);

    // Set grant for byte-based scheduling or remove RWND clamp for time-based
    if (scheduling_mode == "byte") {
      struct rm_grant_info grant_info {};
      err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
      if (err != 0) {
        RM_PRINTF("ERROR: Could not find existing grant for flow FD=%d\n", sockfd);
        return false;
      }
      if (grant_info.ungranted_bytes <= 0) {
        RM_PRINTF("INFO: Cannot activate flow FD=%d because it has no ungranted "
                  "bytes\n",
                  sockfd);
        return true;
      }
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes += epoch_bytes;
      grant_info.grant_done = false;
      err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
      if (err != 0) {
        RM_PRINTF("ERROR: Could not set grant for flow FD=%d, err=%d (%s)\n",
                  sockfd, err, strerror(-err));
        return false;
      }
      RM_PRINTF("INFO: Gave flash grant of %d bytes to flow FD=%d "
                "(queues not updated)\n",
                epoch_bytes, sockfd);
    } else if (scheduling_mode == "time") {
      err = bpf_map_delete_elem(flow_to_rwnd_fd, &flow);
      if (err != 0) {
        RM_PRINTF("WARNING: Could not delete RWND clamp for flow FD=%d, "
                  "err=%d (%s)\n",
                  sockfd, err, strerror(-err));
      }
      RM_PRINTF("INFO: Gave flash grant to flow FD=%d in time-based mode "
                "(queues not updated)\n",
                sockfd);
    }
    // Note: Don't trigger ACK since grant will be in the outgoing request
  } else {
    // This flow doesn't get a flash grant - stays paused in queue
    // It will be activated by normal queue-based scheduling later
    RM_PRINTF("INFO: Port mode - flow FD=%d with remote_port=%u "
              "(not in range [%u, %u)) stays paused, will be activated by "
              "normal scheduling\n",
              sockfd, remote_port, monitor_port_start, port_range_end);
  }

  return true;
}

// This should be called before the real send() to ensure that the correct RWND
// is encoded in the outgoing burst request.
bool handle_send(int sockfd, const void *buf, size_t len) {
  if (new_burst_mode == "port") {
    return handle_send_port_mode(sockfd, buf, len);
  } else if (new_burst_mode == "single") {
    return handle_send_single_mode(sockfd, buf, len);
  } else {
    return handle_send_normal_mode(sockfd, buf, len);
  }
}

// accept() handles the responder side and connect() handles the initiator side.
// On these calls, we register the socket for monitoring and set the CCA to
// BPF_CUBIC. All flows are initially paused. Scheduling takes place when the
// applications transmits a burst request via send().
//
// For some reason, C++ function name mangling does not prevent us from
// overriding accept() and connect(), so we do not need 'extern "C"'.

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  // trunk-ignore(clang-tidy/google-readability-casting)
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
  static auto real_accept = (int (*)(int, struct sockaddr *, socklen_t *))(
      dlsym(RTLD_NEXT, "accept"));
  if (real_accept == nullptr) {
    RM_PRINTF("ERROR: Failed to query dlsym for 'accept': %s\n", dlerror());
    return -1;
  }
  int const fd = real_accept(sockfd, addr, addrlen);
  if (fd == -1) {
    RM_PRINTF("ERROR: Real 'accept' failed\n");
    return fd;
  }
  if (noop_mode) {
    return fd;
  }
  if (check_family(addr) != 0) {
    return fd;
  }
  if (!setup_done) {
    RM_PRINTF(
        "ERROR: Cannot handle 'accept', setup not done, returning FD=%d\n", fd);
    return fd;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    RM_PRINTF("INFO: libratemon_interp not running, 'accept' returning FD=%d\n",
              fd);
    return fd;
  }

  register_fd_for_monitoring(fd);
  RM_PRINTF("INFO: Successful 'accept' for FD=%d, got FD=%d\n", sockfd, fd);
  return fd;
}

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
  // trunk-ignore(clang-tidy/google-readability-casting)
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
  static auto real_connect = (int (*)(int, const struct sockaddr *, socklen_t))(
      dlsym(RTLD_NEXT, "connect"));
  if (real_connect == nullptr) {
    RM_PRINTF("ERROR: Failed to query dlsym for 'connect': %s\n", dlerror());
    return -1;
  }
  int const fd = real_connect(sockfd, addr, addrlen);
  if (fd == -1) {
    RM_PRINTF("ERROR: Real 'connect' failed\n");
    return fd;
  }
  if (noop_mode) {
    return fd;
  }
  if (check_family(addr) != 0) {
    return fd;
  }
  if (!setup_done) {
    RM_PRINTF(
        "ERROR: Cannot handle 'connect', setup not done, returning FD=%d\n",
        fd);
    return fd;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    RM_PRINTF(
        "INFO: libratemon_interp not running, 'connect' returning FD=%d\n", fd);
    return fd;
  }

  register_fd_for_monitoring(fd);
  RM_PRINTF("INFO: Successful 'connect' for FD=%d, got FD=%d\n", sockfd, fd);
  return fd;
}

ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
  static auto real_send = (
      // trunk-ignore(clang-tidy/google-readability-casting)
      // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
      (ssize_t(*)(int, const void *, size_t, int))dlsym(RTLD_NEXT, "send"));
  if (real_send == nullptr) {
    RM_PRINTF("ERROR: Failed to query dlsym for 'send': %s\n", dlerror());
    return -1;
  }
  if (noop_mode) {
    return real_send(sockfd, buf, len, flags);
  }
  // Do any ratemon-specific handling before calling the real send().
  if (setup_done && run) {
    if (!handle_send(sockfd, buf, len)) {
      RM_PRINTF("ERROR: Failed to handle 'send' for FD=%d\n", sockfd);
    }
  } else {
    RM_PRINTF("ERROR: Cannot handle 'send' for FD=%d. setup_done=%d, run=%d\n",
              sockfd, setup_done, run);
  }
  ssize_t const ret = real_send(sockfd, buf, len, flags);
  if (ret == -1) {
    RM_PRINTF("ERROR: Real 'send' failed for FD=%d\n", sockfd);
    return ret;
  }
  RM_PRINTF("INFO: Successful 'send' for FD=%d, sent %zd bytes\n", sockfd, ret);
  return ret;
}

// Get around C++ function name mangling.
extern "C" {
int close(int sockfd) {
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
  static auto real_close = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == nullptr) {
    RM_PRINTF("ERROR: Failed to query dlsym for 'close': %s\n", dlerror());
    return -1;
  }
  int const ret = real_close(sockfd);
  if (ret == -1) {
    RM_PRINTF("ERROR: Real 'close' failed\n");
  } else {
    RM_PRINTF("INFO: Successful 'close' for FD=%d\n", sockfd);
  }
  if (noop_mode) {
    return ret;
  }
  if (!setup_done) {
    RM_PRINTF("ERROR: Cannot handle 'close', setup not done, returning FD=%d\n",
              sockfd);
    return ret;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    RM_PRINTF("INFO: libratemon_interp not running, returning from 'close' for "
              "FD=%d\n",
              sockfd);
    return ret;
  }

  // Remove this FD from all data structures.
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  // trunk-ignore(clang-tidy/clang-diagnostic-error)
  if (fd_to_flow.contains(sockfd)) {
    // Obviously, do this before removing the FD from fd_to_flow.
    remove_flow_from_all_maps(&fd_to_flow[sockfd]);
    // Removing the FD from fd_to_flow triggers it to be (eventually) removed
    // from scheduling.
    rm_flow const flow = fd_to_flow[sockfd];
    rm_flow_key const key = {flow.local_addr, flow.remote_addr, flow.local_port,
                             flow.remote_port};
    auto removed = fd_to_flow.erase(sockfd);
    RM_PRINTF("INFO: Removed FD=%d from fd_to_flow (%ld elements removed)\n",
              sockfd, removed);
    removed = flow_to_fd.erase(key);
    RM_PRINTF("INFO: Removed FD=%d from flow_to_fd (%ld elements removed)\n",
              sockfd, removed);

    // If this is the last flow that we know about and we close it, then assume
    // that we are no longer needed and kill the scheduler thread.
    if (fd_to_flow.empty()) {
      RM_PRINTF("INFO: No more flows remaining, stopping libratemon_interp and "
                "its scheduler thread.\n");
      run = false;
      lock.unlock();
      if (scheduler_thread && scheduler_thread->joinable()) {
        scheduler_thread->join();
        scheduler_thread.reset();
      }
      if (ringbuf_poll_thread && ringbuf_poll_thread->joinable()) {
        ringbuf_poll_thread->join();
        ringbuf_poll_thread.reset();
      }
      return ret;
    }

    // The flow will be removed from the active_fds_queue and paused_fds_queue
    // when the scheduler thread wakes up and processes the next event.
  } else {
    RM_PRINTF("INFO: Ignoring 'close' for FD=%d, not in fd_to_flow\n", sockfd);
  }
  // lock is automatically released when it goes out of scope
  return ret;
}
}

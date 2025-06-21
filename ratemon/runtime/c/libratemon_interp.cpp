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
// Protects writes and reads to active_fds_queue, paused_fds_queue, and
// fd_to_flow.
std::mutex lock_scheduler;
// FDs for flows that are are currently active.
std::queue<std::pair<int, boost::posix_time::ptime>> active_fds_queue;
// FDs for flows that are currently paused (RWND = 0 B);
std::queue<int> paused_fds_queue;
// Maps file descriptor to rm_flow struct.
std::unordered_map<int, struct rm_flow> fd_to_flow;
std::unordered_map<struct rm_flow_key, int> flow_to_fd;
// Maps rm_flow struct to the number of bytes remaining in the flow's message.
// Used to give a partial grant when the flow is about to finish.
std::unordered_map<int, int> flow_to_pending_bytes;
// The next six are scheduled RWND tuning parameters. See ratemon.h for
// parameter documentation.
int max_active_flows = 5;
std::string scheduling_mode = "time"; // or "bytes"
int epoch_us = 10000;
int epoch_bytes = 65536;
// int response_size_bytes = 0;
int idle_timeout_us = 0;
int64_t idle_timeout_ns = 0;
// Ports in this range (inclusive) will be tracked for scheduling.
uint16_t monitor_port_start = 9000;
uint16_t monitor_port_end = 9999;

// Forward declaration so that setup() resolves. Defined for real below.
bool setup();
// Whether setup has been performed.
bool setup_done = setup();

// Used to set entries in flow_to_rwnd.
int zero = 0;
boost::posix_time::seconds one_sec = boost::posix_time::seconds(1);
// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length =
    static_cast<socklen_t>(sizeof(placeholder_cc_info));

// Trigger a pure ACK packet to be send on this FD by calling getsockopt() with
// TCP_CC_INFO. This only works if the flow is using the CCA BPF_CUBIC.
inline void trigger_ack(int fd) {
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
inline int pause_flow(int fd) {
  // Pausing a flow means setting its RWND to 0 B.
  int const err =
      bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &zero, BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Could not pause flow FD=%d, err=%d (%s)\n", fd, err,
              strerror(-err));
    return 0;
  }
  trigger_ack(fd);
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
      paused_fds_queue.push(active_fr.first);
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
  if (scheduling_mode == "byte") {
    // Look up the remaining data in this flow.
    auto pending_bytes = flow_to_pending_bytes.find(fd);
    if (pending_bytes == flow_to_pending_bytes.end()) {
      RM_PRINTF("ERROR: Could not find pending bytes for FD=%d\n", fd);
      return 0;
    }
    // Determine how many bytes to grant.
    int grant_bytes = std::min(epoch_bytes, pending_bytes->second);
    // Decrement the pending bytes by the grant size since the flow will be able
    // to send this much data. We can be confident that that we will not call
    // activate_flow() again until the flow has sent this much data.
    pending_bytes->second -= grant_bytes;
    // Assign the grant.
    int const err = bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd],
                                        &grant_bytes, BPF_ANY);
    if (err != 0) {
      RM_PRINTF("ERROR: Could not set grant for flow FD=%d, err=%d (%s)\n", fd,
                err, strerror(-err));
      return 0;
    }
  } else if (scheduling_mode == "time") {
    // Remove the RWND limit of 0 that has paused the flow.
    int const err = bpf_map_delete_elem(flow_to_rwnd_fd, &fd_to_flow[fd]);
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
    pause_fr = paused_fds_queue.front();
    paused_fds_queue.pop();
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
      paused_fds_queue.push(pause_fr);
      continue;
    }
    RM_PRINTF("INFO: Decided to activating flow FD=%d\n", pause_fr);
    // Randomly jitter the epoch time by +/- 12.5%.
    active_fds_queue.emplace(
        pause_fr,
        now_plus_epoch + boost::posix_time::microseconds(jitter(epoch_us)));
    return activate_flow(pause_fr);
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
  lock_scheduler.lock();
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
    // flows.
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
          RM_PRINTF(
              "WARNING: FD=%d last data time (%lu ns) is in the future compared to our current time (%ld ns) by %ld ns. This is probably due to a super recent sneaky packet arrival since we recorded the current time.\n",
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
          paused_fds_queue.push(active_fr.first);
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
      RM_PRINTF("INFO: In byte-based scheduling mode, using slow check mode\n");
      when = one_sec;
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
  lock_scheduler.unlock();
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
  lock_scheduler.lock();
  for (const auto &pair : fd_to_flow) {
    remove_flow_from_all_maps(&pair.second);
  }
  lock_scheduler.unlock();
  RM_PRINTF("INFO: Scheduler thread ended\n");

  // Need to manually free the ring buffer.
  if (done_flows_rb != nullptr) {
    ring_buffer__free(done_flows_rb);
  }

  if (run) {
    RM_PRINTF("ERROR: Scheduled thread ended before program was signalled to "
              "stop\n");
  }
}

int handle_grant_done(void * /*ctx*/, void *data, size_t data_sz) {
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
  RM_PRINTF("INFO: Flow %s:%u->%s:%u has exhausted its grant\n",
            ipv4_to_string(flow->remote_addr).c_str(), flow->remote_port,
            ipv4_to_string(flow->local_addr).c_str(), flow->local_port);

  // Activate a new flow.
  lock_scheduler.lock();
  rm_flow_key const key = {flow->local_addr, flow->remote_addr,
                           flow->local_port, flow->remote_port};
  auto fd = flow_to_fd.find(key);
  if (fd == flow_to_fd.end()) {
    RM_PRINTF("ERROR: Could not find FD for flow %s:%u->%s:%u\n",
              ipv4_to_string(flow->remote_addr).c_str(), flow->remote_port,
              ipv4_to_string(flow->local_addr).c_str(), flow->local_port);
    lock_scheduler.unlock();
    return 0;
  }
  if (try_pause_one_activate_one(fd->second) == 0) {
    RM_PRINTF("ERROR: Could not pause flow FD=%d and activate another flow\n",
              fd->second);
  }
  lock_scheduler.unlock();
  return 0;
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
// parameters from environment variables and looking up the BPF map
// flow_to_rwnd.
bool setup() {
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
  // if (!read_env_int(RM_RESPONSE_SIZE_KEY, &response_size_bytes)) {
  //   return false;
  // }
  // if (response_size_bytes < 0) {
  //   RM_PRINTF(
  //       "ERROR: Invalid value for '%s'=%d (must be > 0; set = 0 to
  //       disable)\n", RM_RESPONSE_SIZE_KEY, response_size_bytes);
  //   return false;
  // }
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

// Perform initial scheduling for this flow. This should happen during the
// handshake, in either accept() or connect(). The caller must hold
// lock_scheduler.
void initial_scheduling(int fd) {
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
  // Should this flow be active or paused?
  if (static_cast<int>(active_fds_queue.size()) >= max_active_flows) {
    // The max number of flows are active already, so pause this one.
    paused_fds_queue.push(fd);
    pause_flow(fd);
    return;
  }

  // Less than the max number of flows are active, so make this one active.
  boost::posix_time::ptime const now =
      boost::posix_time::microsec_clock::local_time();
  auto awaken = now + boost::posix_time::microseconds(epoch_us) +
                boost::posix_time::microseconds(jitter(epoch_us));
  active_fds_queue.emplace(fd, awaken);
  RM_PRINTF(
      "Initial scheduling for flow FD=%d, awaken in the future by %ld us\n", fd,
      (active_fds_queue.front().second - now).total_microseconds());
  RM_PRINTF("INFO: Allowing new flow FD=%d\n", fd);
  if (active_fds_queue.size() == 1) {
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
    }
    // The above call to timer.expires_from_now() will have cancelled any other
    // pending async_wait(), so we are safe to call async_wait() again.
    timer.async_wait(&timer_callback);
    RM_PRINTF("INFO: First scheduling event\n");
  }

  // If we are using time-based scheduling, then we can immediately set the
  // correct RWND for the flow. We cannot do this for byte-based scheduling
  // because we do not yet have an entry for the flow in pending_bytes. This
  // is because initial_scheduling() is called during the handshake, whereas
  // pending_bytes is initially populated during the burst request, which
  // happens after the handshake.
  if (scheduling_mode == "time") {
    // Activate the flow, but do not trigger an ACK because this is the
    // initial scheduling, which takes place during the handshake. In
    // practice, this call does nothing for time-based scheduling, but it is
    // included for completeness and to get a log indicating the flow was
    // activated.
    if (activate_flow(fd, false /* trigger_ack_on_activate */) == 0) {
      RM_PRINTF("ERROR: Failed to activate flow FD=%d\n", fd);
    }
  }
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
  RM_PRINTF("INFO: Found flow: %s:%u->%s:%u\n",
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
  lock_scheduler.lock();
  fd_to_flow[fd] = flow;
  rm_flow_key const key = {flow.local_addr, flow.remote_addr, flow.local_port,
                           flow.remote_port};
  flow_to_fd[key] = fd;
  // Change the CCA to BPF_CUBIC.
  if (!set_cca(fd, RM_BPF_CUBIC)) {
    lock_scheduler.unlock();
    return;
  }
  // Initial scheduling for this flow.
  initial_scheduling(fd);
  lock_scheduler.unlock();
}

// accept() and connect() are the two entrance points for libratemon_interp.
// accept() handles the responder side and connect() handles the initiator side.
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
  ssize_t const ret = real_send(sockfd, buf, len, flags);
  if (ret == -1) {
    RM_PRINTF("ERROR: Real 'send' failed for FD=%d\n", sockfd);
    return ret;
  }
  if (!setup_done) {
    RM_PRINTF("ERROR: Cannot handle 'send', setup not done, returning FD=%d\n",
              sockfd);
    return ret;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    RM_PRINTF("INFO: libratemon_interp not running, returning from 'send' for "
              "FD=%d\n",
              sockfd);
    return ret;
  }

  // Update the flow_to_keepalive map to indicate that this flow has pending
  // data.
  lock_scheduler.lock();
  auto flow = fd_to_flow.find(sockfd);
  if (flow == fd_to_flow.end()) {
    // We are not tracking this flow, so ignore it.
    RM_PRINTF("INFO: Ignoring 'send' for FD=%d, not in fd_to_flow\n", sockfd);
    lock_scheduler.unlock();
    return ret;
  }
  int one = 1;
  int const err =
      bpf_map_update_elem(flow_to_keepalive_fd, &flow->second, &one, BPF_ANY);
  if (err != 0) {
    RM_PRINTF("ERROR: Failed to update flow_to_keepalive for FD=%d, "
              "flow_to_keepalive_fd=%d, err=%d (%s)\n",
              sockfd, flow_to_keepalive_fd, err, strerror(-err));
    lock_scheduler.unlock();
    return ret;
  }
  RM_PRINTF("INFO: Updated flow_to_keepalive for FD=%d\n", sockfd);

  // Parse buf, which should contain 2 ints. The first is the response size,
  // which is what we want.
  if (len == 2 * sizeof(int)) {
    // trunk-ignore(clang-tidy/google-readability-casting)
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
    int *buf_int = (int *)buf;
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-bounds-pointer-arithmetic)
    int const bytes = buf_int[0];
    // int sender_wait_us = buf_int[1];
    flow_to_pending_bytes[sockfd] = bytes;
    RM_PRINTF("INFO: FD=%d has %d bytes pending\n", sockfd, bytes);
  }

  RM_PRINTF("INFO: Successful 'send' for FD=%d, sent %zd bytes\n", sockfd, ret);
  lock_scheduler.unlock();
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
  lock_scheduler.lock();
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
      lock_scheduler.unlock();
      if (scheduler_thread && scheduler_thread->joinable()) {
        scheduler_thread->join();
        scheduler_thread.reset();
      }
      return ret;
    }

    // The flow will be removed from the active_fds_queue and paused_fds_queue
    // when the scheduler thread wakes up and processes the next event.
  } else {
    RM_PRINTF("INFO: Ignoring 'close' for FD=%d, not in fd_to_flow\n", sockfd);
  }
  lock_scheduler.unlock();
  return ret;
}
}

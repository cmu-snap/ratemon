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
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <experimental/random>
#include <glog/logging.h>
#include <linux/bpf.h>
#include <linux/inet_diag.h>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <queue>
#include <shared_mutex>
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

// Used to signal the scheduler thread to end. Atomic for thread-safe access.
std::atomic<bool> run{true};
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
// Accessed by both the scheduler thread (free) and the ringbuf poll thread
// (poll), so use std::atomic for safe cross-thread visibility.
std::atomic<struct ring_buffer *> done_flows_rb{nullptr};
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
// Per-host index: remote_addr -> list of FDs from that host.
// Maintained alongside fd_to_flow so handle_send_single_mode() can skip
// scanning all flows. Protected by lock_scheduler.
std::unordered_map<uint32_t, std::vector<int>> addr_to_fds;
// The next six are scheduled RWND tuning parameters. See ratemon.h for
// parameter documentation.
int max_active_flows = 5;
std::string scheduling_mode = "byte";  // or "time"
std::string new_burst_mode = "normal"; // or "port" or "single"
std::string single_request_policy =
    "normal"; // or "pregrant" (only applies when new_burst_mode="single")
int epoch_us = 10000;
int epoch_bytes = 65536;
int idle_timeout_us = -1;
int64_t idle_timeout_ns = -1;
// Burst tracking for single_request_pregrant policy
int current_burst_number = 0;
int burst_flows_remaining = 0;
bool pregrant_done = false;
// True after the last flow in a burst finishes (burst_flows_remaining hits 0)
// and before a new burst request arrives via send(). If handle_grant_done fires
// while this is true, it is a spurious/stale event. Protected by
// lock_scheduler.
bool between_bursts = false;
// Deferred burst bytes for pregrant mode. When should_schedule=false,
// handle_send_single_mode() caches burst info per-host instead of updating
// each flow's BPF maps. Applied lazily during
// activate_flow()/handle_grant_done(). Maps remote_addr -> {bytes,
// burst_number}. Protected by lock_scheduler.
std::unordered_map<uint32_t, std::pair<int, int>> pending_burst_info;
// Per-flow: last burst number whose deferred bytes were applied to BPF maps.
// Protected by lock_scheduler.
std::unordered_map<int, int> fd_last_applied_burst;
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
  VLOG(1) << "Triggering ACK for flow FD=" << fd;
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

// Check if the BPF program has signaled an error condition. If so, log the
// error and exit with a fatal error. This should be called after every
// successful bpf_map_lookup_elem that retrieves grant_info.
inline void check_bpf_error(const struct rm_grant_info &grant_info,
                            const char *context) {
  if (grant_info.bpf_experienced_error) {
    LOG(FATAL) << "BPF program experienced an error (detected in " << context
               << "). Exiting.";
    std::exit(EXIT_FAILURE);
  }
}

// Pause this flow. Return the number of flows that were paused.
inline int pause_flow(int fd, bool trigger_ack_on_pause = true) {
  if (scheduling_mode == "byte") {
    VLOG(1) << "Cannot pause flow FD=" << fd
            << " in byte-based scheduling mode";
    return 1;
  }
  // Pausing a flow means setting its RWND to 0 B.
  struct rm_grant_info grant_info {};
  // Need to look up instead of simply overwriting because unacked_bytes must be
  // preserved.
  int err = bpf_map_lookup_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info);
  if (err != 0) {
    // No existing rm_grant_info, so fill in new info.
    LOG(ERROR) << "Could not find existing grant for flow FD=" << fd;
    return 0;
  }
  check_bpf_error(grant_info, "pause_flow");
  grant_info.override_rwnd_bytes = 0;
  err = bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info,
                            BPF_ANY);
  if (err != 0) {
    LOG(ERROR) << "Could not pause flow FD=" << fd << ", err=" << err << " ("
               << strerror(-err) << ")";
    return 0;
  }
  if (trigger_ack_on_pause) {
    // Trigger an ACK to be sent on this flow.
    VLOG(1) << "Triggering ACK for paused flow FD=" << fd;
    trigger_ack(fd);
  }
  VLOG(1) << "Paused flow FD=" << fd;
  return 1;
}

// Find and pause the flow. Return the number of flows that were paused.
int try_find_and_pause(int fd) {
  VLOG(1) << "Trying to pause flow FD=" << fd;
  // Temporary variable for storing the front of active_fds_queue.
  std::pair<int, boost::posix_time::ptime> active_fr;
  int active_idx = 0;
  // Check all entries in active_fds_queue once, until we find the one we are
  // looking for.
  while (active_idx < static_cast<int>(active_fds_queue.size())) {
    active_fr = active_fds_queue.front();
    VLOG(1) << "Checking active flow " << active_fr.first << " at index "
            << active_idx;
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

// Check whether a flow has deferred burst bytes that haven't been applied yet.
// Caller must hold lock_scheduler (at least shared).
inline bool has_pending_burst_bytes(int fd) {
  auto fit = fd_to_flow.find(fd);
  if (fit == fd_to_flow.end())
    return false;
  auto pbi = pending_burst_info.find(fit->second.remote_addr);
  if (pbi == pending_burst_info.end())
    return false;
  auto applied = fd_last_applied_burst.find(fd);
  return (applied == fd_last_applied_burst.end() ||
          applied->second < pbi->second.second);
}

// Add a flow to the flow_to_keepalive map. Must hold lock_scheduler before
// calling this function.
bool set_keepalive(int sockfd) {
  auto flow = fd_to_flow.find(sockfd);
  if (flow == fd_to_flow.end()) {
    // We are not tracking this flow, so ignore it.
    VLOG(1) << "Ignoring 'send' for FD=" << sockfd << ", not in fd_to_flow";
    return false;
  }
  int one = 1;
  int const err =
      bpf_map_update_elem(flow_to_keepalive_fd, &flow->second, &one, BPF_ANY);
  if (err != 0) {
    LOG(ERROR) << "Failed to update flow_to_keepalive for FD=" << sockfd
               << ", flow_to_keepalive_fd=" << flow_to_keepalive_fd
               << ", err=" << err << " (" << strerror(-err) << ")";
    return false;
  }
  VLOG(1) << "Updated flow_to_keepalive for FD=" << sockfd;
  return true;
}

// Apply deferred burst bytes to a flow's grant_info if needed.
// Modifies grant_info in place (caller must write back to BPF if it returns
// true). Also sets keepalive in BPF for this flow.
// Returns true if deferred bytes were applied.
// Caller must hold lock_scheduler (unique).
inline bool apply_deferred_burst_bytes(int fd,
                                       struct rm_grant_info &grant_info) {
  auto fit = fd_to_flow.find(fd);
  if (fit == fd_to_flow.end())
    return false;
  auto pbi = pending_burst_info.find(fit->second.remote_addr);
  if (pbi == pending_burst_info.end())
    return false;
  int burst_num = pbi->second.second;
  auto &last_applied = fd_last_applied_burst[fd];
  if (last_applied >= burst_num)
    return false;

  grant_info.ungranted_bytes += pbi->second.first;
  last_applied = burst_num;
  set_keepalive(fd);

  VLOG(1) << "Applied deferred burst bytes (" << pbi->second.first
          << ") for FD=" << fd << ", burst=" << burst_num
          << ", new ungranted_bytes=" << grant_info.ungranted_bytes;
  return true;
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
      LOG(ERROR) << "Could not find existing grant for flow FD=" << fd;
      return 0;
    }
    check_bpf_error(grant_info, "activate_flow");
    // Apply any deferred burst bytes before checking ungranted_bytes.
    apply_deferred_burst_bytes(fd, grant_info);
    if (grant_info.ungranted_bytes <= 0) {
      VLOG(1) << "Cannot activate flow FD=" << fd
              << " because it has no ungranted bytes";
      return 0;
    }
    grant_info.override_rwnd_bytes = 0xFFFFFFFF;
    grant_info.new_grant_bytes += epoch_bytes;
    grant_info.grant_done = false;
    grant_info.is_pregrant = false;
    // Write the new grant info into the map.
    err = bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &grant_info,
                              BPF_ANY);
    if (err != 0) {
      LOG(ERROR) << "Could not set grant for flow FD=" << fd << ", err=" << err
                 << " (" << strerror(-err) << ")";
      return 0;
    }
    VLOG(1) << "Activated flow FD=" << fd << " with grant of " << epoch_bytes
            << " bytes";
  } else if (scheduling_mode == "time") {
    // Remove the RWND limit of 0 that has paused the flow.
    err = bpf_map_delete_elem(flow_to_rwnd_fd, &fd_to_flow[fd]);
    if (err != 0) {
      LOG(WARNING) << "Could not delete RWND clamp for flow FD=" << fd
                   << ", err=" << err << " (" << strerror(-err)
                   << "). The flow might not have been clamped.";
    }
  }
  if (trigger_ack_on_activate) {
    trigger_ack(fd);
  }
  VLOG(1) << "Activated FD=" << fd;
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
      VLOG(1) << "Skipping activating FD=" << pause_fr << ", flow closed";
      continue;
    }
    // If this flow is not in the flow_to_keepalive map (bpf_map_lookup_elem()
    // returns negative error code when the flow is not found), then it has no
    // pending data and should be skipped — unless it has deferred burst bytes
    // that haven't been applied yet.
    int const err = bpf_map_lookup_elem(flow_to_keepalive_fd,
                                        &fd_to_flow[pause_fr], &dummy);
    VLOG(1) << "Checking flow FD=" << pause_fr << ", dummy=" << dummy
            << ", err=" << err;
    if (err != 0 && !has_pending_burst_bytes(pause_fr)) {
      VLOG(1) << "Skipping activating FD=" << pause_fr << ", no pending data";
      paused_fds_queue.enqueue(pause_fr);
      continue;
    }

    VLOG(1) << "Trying to activate flow FD=" << pause_fr;
    if (activate_flow(pause_fr, true) == 0) {
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
    LOG(ERROR) << "Could not find and/or pause flow FD=" << fd;
    // TODO: return 0;
  }
  // Then find one flow in paused_fds_queue to restart.
  int const num_activated = try_activate_one();
  if (num_activated == 0) {
    LOG(ERROR) << "Could not activate a flow after pausing FD=" << fd;
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
  VLOG(1) << "In timer_callback";

  // 0. Perform validity checks.
  // If an error (such as a cancellation) triggered this callback, then abort
  // immediately. Do not set another timer.
  if (error) {
    LOG(ERROR) << "timer_callback error: " << error.message().c_str();
    return;
  }
  // If the program has been signalled to stop, then exit. Do not set another
  // timer.
  if (!run) {
    VLOG(1) << "Program signalled to exit";
    return;
  }
  // If setup has not been performed yet, then we cannot perform scheduling.
  // Otherwise, revert to slow check mode.
  if (!setup_done) {
    LOG(ERROR) << "Cannot execute timer callback, setup not done";
    if (timer.expires_from_now(one_sec) != 0U) {
      LOG(ERROR) << "Timer unexpectedly cancelled (1)";
    }
    timer.async_wait(&timer_callback);
    return;
  }
  // Check that relevant parameters have been set. Otherwise, revert to slow
  // check mode.
  if ((max_active_flows == 0U) || (epoch_us == 0U) || (flow_to_rwnd_fd == 0) ||
      (flow_to_win_scale_fd == 0) || (flow_to_last_data_time_fd == 0) ||
      (flow_to_keepalive_fd == 0)) {
    LOG(ERROR) << "cannot continue, invalid max_active_flows="
               << max_active_flows << ", epoch_us=" << epoch_us
               << ", flow_to_rwnd_fd=" << flow_to_rwnd_fd
               << ", flow_to_win_scale_fd=" << flow_to_win_scale_fd
               << ", flow_to_last_data_time_fd=" << flow_to_last_data_time_fd
               << ", or flow_to_keepalive_fd=" << flow_to_keepalive_fd;
    if (timer.expires_from_now(one_sec) != 0U) {
      LOG(ERROR) << "Timer unexpectedly cancelled (2)";
    }
    timer.async_wait(&timer_callback);
    return;
  }

  // It is now safe to perform scheduling.
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  VLOG(1) << "Performing scheduling. active=" << active_fds_queue.size()
          << ", paused=" << paused_fds_queue.size();

  // Temporary variable for storing the front of active_fds_queue.
  std::pair<int, boost::posix_time::ptime> active_fr;
  // Size of active_fds_queue.
  int active_size = 0;
  // Vector of active flows that we plan to pause.
  std::vector<int> to_pause;
  // Current kernel time (since boot).
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int64_t const ktime_now_ns =
      static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
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
    LOG(FATAL) << "active_fds_queue.size()=" << active_fds_queue.size()
               << " is larger than max_active_flows=" << max_active_flows;
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
    if (idle_timeout_ns >= 0 && !paused_fds_queue.empty()) {
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
          LOG(WARNING) << "FD=" << active_fr.first << " last data time ("
                       << last_data_time_ns
                       << " ns) is in the future compared to our current time ("
                       << ktime_now_ns << " ns) by "
                       << last_data_time_ns - ktime_now_ns
                       << " ns. This is probably due to a super recent sneaky "
                          "packet arrival since we recorded the current time.";
        }

        idle_ns = ktime_now_ns - last_data_time_ns;
        VLOG(1) << "FD=" << active_fr.first << " now: " << ktime_now_ns
                << " ns, last data time: " << last_data_time_ns << " ns";
        VLOG(1) << "FD=" << active_fr.first << " idle has been idle for "
                << idle_ns << " ns. timeout is " << idle_timeout_ns << " ns";
        // If the flow has been idle for longer than the idle timeout,
        // handle it based on the scheduling policy.
        if (idle_ns >= idle_timeout_ns) {
          if (single_request_policy == "pregrant") {
            // In pregrant mode, an idle active flow experienced a tail loss and
            // is waiting for the retransmission timeout to fire. Give it more
            // grant so it can send more packets and get a triple duk-ACK.
            VLOG(1) << "FD=" << active_fr.first
                    << " idle timeout in pregrant mode, giving extra grant "
                       "instead of pausing";
            activate_flow(active_fr.first, true);
            // Keep the flow in the active queue (do not pause it).
          } else {
            // Default behavior: pause the idle flow. We pause the flow
            // *before* activating a replacement flow because it is by
            // definition not sending data, so we do not risk causing a
            // drop in utilization by pausing it immediately.
            VLOG(1) << "Pausing FD=" << active_fr.first
                    << " due to idle timeout";
            // Remove the flow from flow_to_keepalive, signalling that it
            // no longer has pending demand.
            int const err = bpf_map_delete_elem(flow_to_keepalive_fd,
                                                &fd_to_flow[active_fr.first]);
            if (err != 0) {
              LOG(ERROR) << "Could not delete flow FD=" << active_fr.first
                         << " from keepalive map, err=" << err << " ("
                         << strerror(-err) << ")";
            }
            paused_fds_queue.enqueue(active_fr.first);
            pause_flow(active_fr.first);
            continue;
          }
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
        VLOG(1) << "Reactivated FD=" << active_fr.first;
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

  VLOG(1) << "Flows in to_pause:";
  for (const auto &fd : to_pause) {
    VLOG(1) << "    " << fd;
  }

  // 2) Activate flows. Now we can calculate how many flows to activate to reach
  // full capacity. This value is the existing free capacity plus the number of
  // flows we intend to pause. The important part here is that we only look at
  // as many entries in paused_fds_queue as needed.
  int const num_to_activate = max_active_flows -
                              static_cast<int>(active_fds_queue.size()) +
                              static_cast<int>(to_pause.size());
  VLOG(1) << "Attempting to activate " << num_to_activate << " flows";
  int num_activated = 0;
  for (int i = 0; i < num_to_activate; ++i) {
    num_activated += try_activate_one();
  }
  if (num_activated != num_to_activate) {
    LOG(ERROR) << "Could not activate as many flows as requested, activated="
               << num_activated << ", requested=" << num_to_activate;
    // If we could not activate all requested flows, then we will have to pause
    // some of the active flows.
  } else {
    VLOG(1) << "Activated " << num_activated << " flows";
  }

  // 3) Pause flows. We need to recalculate the number of flows to pause because
  // we may not have been able to activate as many flows as planned. Recall that
  // it is alright to iterate through all of active_fds_queue.
  VLOG(1) << "active_fds_queue.size()=" << active_fds_queue.size()
          << ", paused_fds_queue.size()=" << paused_fds_queue.size()
          << ", max_active_flows=" << max_active_flows;
  int const num_to_pause =
      std::max(0, static_cast<int>(active_fds_queue.size()) - max_active_flows);
  DCHECK(num_to_pause <= static_cast<int>(to_pause.size()));
  // For each flow that we are supposed to pause, advance through
  // active_fds_queue until we find it.
  VLOG(1) << "Pausing " << num_to_pause
          << " flows, to_pause contains: " << to_pause.size();
  active_size = static_cast<int>(active_fds_queue.size());
  for (int i = 0; i < num_to_pause; ++i) {
    try_find_and_pause(to_pause[i]);
  }

  // 4) Check invariants.
  // Cannot have more than the max number of active flows.
  DCHECK(static_cast<int>(active_fds_queue.size()) <= max_active_flows);

  // 5) Calculate when the next timer should expire.
  boost::posix_time::time_duration when;
  if (active_fds_queue.empty()) {
    // If there are no flows, revert to slow check mode.
    VLOG(1) << "No flows remaining, reverting to slow check mode";
    when = one_sec;
  } else {
    auto const next_epoch_us =
        (active_fds_queue.front().second - now).total_microseconds();
    if (scheduling_mode == "byte") {
      if (idle_timeout_ns < 0) {
        // If we are not using idle timeout mode...
        VLOG(1) << "In byte-based scheduling mode but no idle timeout, falling "
                   "back to slow check mode";
        when = one_sec;
      } else {
        VLOG(1) << "In byte-based scheduling mode, scheduling timer for next "
                   "idle timeout";
        when = boost::posix_time::microsec(idle_timeout_us);
      }
    } else if (idle_timeout_ns < 0) {
      // If we are not using idle timeout mode...
      VLOG(1) << "No idle timeout, scheduling timer for next epoch end";
      when = boost::posix_time::microsec(next_epoch_us);
    } else if (idle_timeout_us < next_epoch_us) {
      // If we are using idle timeout mode...
      VLOG(1) << "Scheduling timer for next idle timeout, sooner than next "
                 "epoch end";
      when = boost::posix_time::microsec(idle_timeout_us);
    } else {
      VLOG(1) << "epoch_us=" << epoch_us
              << ", idle_timeout_us=" << idle_timeout_us
              << ", next_epoch_us=" << next_epoch_us;
      VLOG(1)
          << "Scheduling timer for next epoch end, sooner than idle timeout";
      when = boost::posix_time::microsec(next_epoch_us);
    }
  }

  // 6) Start the next timer.
  if (timer.expires_from_now(when) != 0U) {
    LOG(ERROR) << "Timer unexpectedly cancelled (3)";
  }
  timer.async_wait(&timer_callback);
  // lock is automatically released when it goes out of scope
  VLOG(1) << "Sleeping until next event in " << when.total_microseconds()
          << " us";
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
  VLOG(1) << "Scheduler thread started";
  if (timer.expires_from_now(one_sec) != 0U) {
    LOG(ERROR) << "Timer unexpectedly cancelled (4)";
  }

  timer.async_wait(&timer_callback);
  VLOG(1) << "Scheduler thread initial sleep";
  // Execute the configured events, until there are no more events to execute.
  io.run();

  // Clean up all flows.
  {
    std::unique_lock<std::shared_mutex> lock(lock_scheduler);
    for (const auto &pair : fd_to_flow) {
      remove_flow_from_all_maps(&pair.second);
    }
  }
  VLOG(1) << "Scheduler thread ended";

  if (run) {
    LOG(ERROR) << "Scheduler thread ended before program was signalled to stop";
  }

  // Signal the ringbuf poll thread to stop. The poll thread checks `run`
  // each iteration and will exit. Do NOT free or null out done_flows_rb
  // here — the poll thread may still be inside ring_buffer__poll(), and
  // freeing the ring buffer would NULL out its internal consumer_pos
  // pointer mid-use, causing a SEGV. ring_buffer__free() is called in
  // sigint_handler() after the poll thread has been joined.
  run = false;

  VLOG(1) << "Scheduler thread ended";
}

int handle_grant_done(void * /*ctx*/, void *data, size_t data_sz) {
  VLOG(1) << "In handle_grant_done, data_sz=" << data_sz;
  if (!setup_done) {
    LOG(ERROR) << "Cannot handle grant done, setup not done";
    return 0;
  }

  // Diagnostic: if we are between bursts (all flows finished the previous
  // burst and no new burst request has arrived yet), this grant-done event may
  // be from a pregrant being consumed or a straggler finishing. Log it but do
  // NOT return early — the flow still needs to be paused and replaced.
  if (between_bursts) {
    LOG(WARNING)
        << "grant_done while between_bursts=true (burst_flows_remaining="
        << burst_flows_remaining
        << ", current_burst_number=" << current_burst_number
        << ", pregrant_done=" << static_cast<int>(pregrant_done) << ")";
  }

  if (scheduling_mode != "byte") {
    return 0;
  }
  if (data_sz != sizeof(struct rm_flow)) {
    LOG(ERROR) << "Invalid data size " << data_sz << ", expected "
               << sizeof(struct rm_flow);
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
    LOG(ERROR) << "Could not find FD for flow "
               << ipv4_to_string(flow->remote_addr).c_str() << ":"
               << flow->remote_port << "->"
               << ipv4_to_string(flow->local_addr).c_str() << ":"
               << flow->local_port;
    return 0;
  }
  VLOG(1) << "Flow FD=" << fd->second << " has exhausted its grant";

  // Check if this flow completed its entire burst and should be pregranted.
  // In pregrant mode, when a flow finishes its burst, we give the pregrant
  // directly to THAT flow (not the next activated one). This avoids the problem
  // where we try to pregrant the next flow to activate but there are no paused
  // flows to activate (all remaining flows are already active in the straggler
  // phase).
  if (single_request_policy == "pregrant" && new_burst_mode == "single") {
    struct rm_grant_info grant_info {};
    int err = bpf_map_lookup_elem(flow_to_rwnd_fd, flow, &grant_info);
    if (err == 0) {
      check_bpf_error(grant_info, "handle_grant_done");
      // Apply any deferred burst bytes before checking burst completion.
      bool needs_write = apply_deferred_burst_bytes(fd->second, grant_info);
      if (grant_info.ungranted_bytes <= 0) {
        // This flow completed its burst.
        burst_flows_remaining--;
        if (burst_flows_remaining <= 0) {
          between_bursts = true;
          VLOG(1) << "All flows completed burst " << current_burst_number
                  << ", between_bursts=true";
        }
        VLOG(1) << "Flow FD=" << fd->second << " completed burst, "
                << burst_flows_remaining << " flows remaining";

        // Pregrant the last max_active_flows to finish. Once
        // burst_flows_remaining drops below max_active_flows, each completing
        // flow gets a pregrant for the next burst directly.
        if (burst_flows_remaining < max_active_flows) {
          // Give pregrant directly to THIS flow. The BPF will extend
          // both rwnd_end_seq (opening the RWND for the next burst) and
          // grant_end_seq (so the BPF tracks when the pregranted window
          // is consumed).
          grant_info.new_grant_bytes = epoch_bytes;
          grant_info.is_pregrant = true;
          // Reset grant_done so the BPF can properly track when this pregrant
          // is consumed. The BPF will not re-submit to done_flows on the same
          // packet that processes a new grant (processed_new_grant guard).
          grant_info.grant_done = false;
          err =
              bpf_map_update_elem(flow_to_rwnd_fd, flow, &grant_info, BPF_ANY);
          if (err != 0) {
            LOG(ERROR) << "Could not set pregrant for flow FD=" << fd->second
                       << ", err=" << err << " (" << strerror(-err) << ")";
          } else {
            VLOG(1) << "Pregranted flow FD=" << fd->second << " directly ("
                    << burst_flows_remaining << " burst flows remaining)";
            // Trigger an ACK so the BPF processes the pregrant and opens
            // the RWND for the next burst.
            trigger_ack(fd->second);
          }
          needs_write = false; // already wrote the updated grant_info
        }
      }
      if (needs_write) {
        // Write the updated grant_info back to the map.
        err = bpf_map_update_elem(flow_to_rwnd_fd, flow, &grant_info, BPF_ANY);
        if (err != 0) {
          LOG(ERROR) << "Could not write updated grant_info for flow FD="
                     << fd->second << ", err=" << err << " (" << strerror(-err)
                     << ")";
        }
      }
    }
  }

  // Pause this flow and activate another (never pregrant on activation).
  if (try_pause_one_activate_one(fd->second) == 0) {
    LOG(ERROR) << "Could not pause flow FD=" << fd->second
               << " and activate another flow";
  }
  return 0;
}

// Function to poll the done_flows ringbuffer. Only used in byte-based
// scheduling mode.
void ringbuf_poll_func() {
  VLOG(1) << "Ringbuf poll thread started";
  while (run) {
    // Safety check: if done_flows_rb was set to nullptr (shouldn't happen
    // during normal operation, but guard against it), stop polling.
    struct ring_buffer *rb = done_flows_rb.load(std::memory_order_acquire);
    if (rb == nullptr) {
      break;
    }
    int const ret = ring_buffer__poll(rb, 100 /* timeout ms */);
    if (ret < 0 && ret != -EINTR) {
      LOG(ERROR) << "ring_buffer__poll returned " << ret;
      break;
    }
  }
  VLOG(1) << "Ringbuf poll thread ended";
}

// Catch SIGINT and trigger the scheduler thread and timer to end.
void sigint_handler(int signum) {
  switch (signum) {
  case SIGINT: {
    VLOG(1) << "Caught SIGINT";
    run = false;
    // Join the ringbuf poll thread FIRST, before the scheduler thread.
    // The scheduler thread does NOT free or null out done_flows_rb — it
    // only sets run=false to signal the poll thread. We free the ring
    // buffer here after the poll thread has fully exited, so there is
    // no race with ring_buffer__poll() internals.
    if (ringbuf_poll_thread &&
        std::this_thread::get_id() == ringbuf_poll_thread->get_id()) {
      LOG(WARNING) << "Caught SIGINT in the ringbuf poll thread. Should this "
                      "have happened?";
    } else if (ringbuf_poll_thread && ringbuf_poll_thread->joinable()) {
      ringbuf_poll_thread->join();
      ringbuf_poll_thread.reset();
    }
    // Now join the scheduler thread.
    if (scheduler_thread &&
        std::this_thread::get_id() == scheduler_thread->get_id()) {
      LOG(WARNING) << "Caught SIGINT in the scheduler thread. Should this have "
                      "happened?";
    } else if (scheduler_thread && scheduler_thread->joinable()) {
      scheduler_thread->join();
      scheduler_thread.reset();
    }
    // Now that both threads are done, safely free the ring buffer.
    struct ring_buffer *rb = done_flows_rb.exchange(nullptr);
    if (rb != nullptr) {
      ring_buffer__free(rb);
    }
    VLOG(1) << "Resetting old SIGINT handler";
    sigaction(SIGINT, &oldact, nullptr);
    break;
  }
  default:
    LOG(ERROR) << "Caught signal " << signum;
    break;
  }
  VLOG(1) << "Re-raising signal " << signum;
  raise(signum);
}

// Read an environment variable as an integer.
bool read_env_int(const char *key, volatile int *dest, bool allow_zero = false,
                  bool allow_neg = false) {
  char *val_str = getenv(key);
  if (val_str == nullptr) {
    LOG(ERROR) << "Failed to query environment variable '" << key << "'";
    return false;
  }
  int const val_int = atoi(val_str);
  if (!allow_zero and val_int == 0) {
    LOG(ERROR) << "Invalid value for '" << key << "'=" << val_int
               << " (must be != 0)";
    return false;
  }
  if (!allow_neg and val_int < 0) {
    LOG(ERROR) << "Invalid value for '" << key << "'=" << val_int
               << " (must be > 0)";
    return false;
  }
  *dest = val_int;
  return true;
}

// Read an environment variable as a string.
bool read_env_string(const char *key, std::string &dest) {
  char *val_str = getenv(key);
  if (val_str == nullptr) {
    LOG(ERROR) << "Failed to query environment variable '" << key << "'";
    return false;
  }
  // Check that the string is not empty.
  if (strlen(val_str) == 0) {
    LOG(ERROR) << "Invalid value for '" << key << "'='" << val_str
               << "' (must be non-empty)";
    return false;
  }
  dest = std::string(val_str);
  return true;
}

// Perform setup (only once for all flows in this process), such as reading
// parameters from environment variables and setting up BPF maps.
bool setup() {
  google::InitGoogleLogging("libratemon_interp");
  google::InstallFailureSignalHandler();
  // Default to WARNING level. Override at runtime with GLOG_minloglevel=0 for
  // INFO or GLOG_v=1 for VLOG messages.
  FLAGS_minloglevel = 1;

  // Check if no-op mode is enabled first
  const char *noop_env = std::getenv("RM_NOOP_MODE");
  if (noop_env != nullptr && std::string(noop_env) == "yes") {
    noop_mode = true;
    LOG(WARNING) << "Running in no-op mode. All functions will delegate to "
                    "kernel implementations";
    // In no-op mode we still do all other setup, but later we do not perform
    // any ratemon operations.
  }

  // Parameter setup.

  VLOG(1) << "Performing setup";
  // Read environment variables with parameters.
  if (!read_env_int(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows)) {
    return false;
  }
  if (!read_env_string(RM_SCHEDILING_MODE_KEY, scheduling_mode)) {
    return false;
  }
  if (scheduling_mode != "time" && scheduling_mode != "byte") {
    LOG(ERROR) << "Invalid value for '" << RM_SCHEDILING_MODE_KEY << "'='"
               << scheduling_mode.c_str() << "' (must be 'time' or 'byte')";
    return false;
  }

  if (!read_env_string(RM_NEW_BURST_MODE_KEY, new_burst_mode)) {
    // Default to "normal" if not specified
    new_burst_mode = "normal";
    VLOG(1) << "RM_NEW_BURST_MODE not set, defaulting to 'normal'";
  }
  VLOG(1) << "new_burst_mode=" << new_burst_mode.c_str();
  if (new_burst_mode != "normal" && new_burst_mode != "port" &&
      new_burst_mode != "single") {
    LOG(ERROR) << "Invalid new_burst_mode=" << new_burst_mode.c_str()
               << ", must be 'normal', 'port', or 'single'";
    return false;
  }

  // Read and validate single_request_policy
  if (!read_env_string(RM_SINGLE_REQUEST_POLICY_KEY, single_request_policy)) {
    // Default to "normal" if not specified
    single_request_policy = "normal";
    VLOG(1) << "RM_SINGLE_REQUEST_POLICY not set, defaulting to 'normal'";
  }
  VLOG(1) << "single_request_policy=" << single_request_policy.c_str();
  if (single_request_policy != "normal" &&
      single_request_policy != "pregrant") {
    LOG(ERROR) << "Invalid single_request_policy="
               << single_request_policy.c_str()
               << ", must be 'normal' or 'pregrant'";
    return false;
  }

  if (!read_env_int(RM_EPOCH_US_KEY, &epoch_us)) {
    return false;
  }
  if (!read_env_int(RM_EPOCH_BYTES_KEY, &epoch_bytes)) {
    return false;
  }
  if (epoch_bytes < 1) {
    LOG(ERROR) << "Invalid value for '" << RM_EPOCH_BYTES_KEY
               << "'=" << epoch_bytes << " (must be > 0)";
    return false;
  }
  if (!read_env_int(RM_IDLE_TIMEOUT_US_KEY, &idle_timeout_us,
                    true /* allow_zero */, true /* allow_neg */)) {
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
    LOG(ERROR) << "Failed to get FD for 'flow_to_rwnd' from path: '"
               << RM_FLOW_TO_RWND_PIN_PATH << "', err=" << -err << " : "
               << strerror(-err);
    return false;
  }
  flow_to_rwnd_fd = err;

  // Look up the FD for the flow_to_win_scale map. We do not need the BPF
  // skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_WIN_SCALE_PIN_PATH);
  if (err < 0) {
    LOG(ERROR) << "Failed to get FD for 'flow_to_win_scale' from path: '"
               << RM_FLOW_TO_WIN_SCALE_PIN_PATH << "', err=" << -err << " : "
               << strerror(-err);
    return false;
  }
  flow_to_win_scale_fd = err;

  // Look up the FD for the flow_to_last_data_time_ns map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH);
  if (err < 0) {
    LOG(ERROR)
        << "Failed to get FD for 'flow_to_last_data_time_ns' from path: '"
        << RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH << "', err=" << -err << " : "
        << strerror(-err);
    return false;
  }
  flow_to_last_data_time_fd = err;

  // Look up the FD for the flow_to_keepalive map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_KEEPALIVE_PIN_PATH);
  if (err < 0) {
    LOG(ERROR) << "Failed to get FD for 'flow_to_keepalive' from path: '"
               << RM_FLOW_TO_KEEPALIVE_PIN_PATH << "', err=" << -err << " : "
               << strerror(-err);
    return false;
  }
  flow_to_keepalive_fd = err;

  if (scheduling_mode == "byte") {
    // Look up the FD for the done_flows ringbuf. We do not need the BPF
    // skeleton for this.
    err = bpf_obj_get(RM_DONE_FLOWS_PIN_PATH);
    if (err < 0) {
      LOG(ERROR) << "Failed to get FD for 'done_flows' from path: '"
                 << RM_DONE_FLOWS_PIN_PATH << "', err=" << -err << " : "
                 << strerror(-err);
      return false;
    }
    // Use the ringbuf fd to create a new userspace ringbuf instance. Note that
    // this must be freed with ring_buffer__free(). It also must be freed on
    // setup() failure, but we create the ringbuffer last so this is not an
    // issue.
    done_flows_rb = ring_buffer__new(err, handle_grant_done, nullptr, nullptr);
    if (done_flows_rb == nullptr) {
      LOG(ERROR) << "Failed to create ring buffer";
      return false;
    }
    ringbuf_poll_thread.emplace(ringbuf_poll_func);
    VLOG(1) << "Successfully created ring buffer for byte-based scheduling";
  }

  // Catch SIGINT to end the program.
  struct sigaction action {};
  action.sa_handler = sigint_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, &oldact);

  // Launch the scheduler thread.
  scheduler_thread.emplace(thread_func);

  VLOG(1) << "Setup complete! max_active_flows=" << max_active_flows
          << ", epoch_us=" << epoch_us
          << ", idle_timeout_ns=" << idle_timeout_ns
          << ", monitor_port_start=" << monitor_port_start
          << ", monitor_port_end=" << monitor_port_end;
  return true;
}

// Fill in the four-tuple for this socket.
bool get_flow(int fd, struct rm_flow *flow) {
  if (flow == nullptr) {
    LOG(ERROR) << "Flow pointer is null";
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
    LOG(ERROR) << "Failed to call 'getsockname'";
    return false;
  }
  struct sockaddr_in remote_addr {};
  socklen_t remote_addr_len = sizeof(remote_addr);
  // Get the peer's (i.e., the remote) IP and port.
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-reinterpret-cast)
  if (getpeername(fd, reinterpret_cast<struct sockaddr *>(&remote_addr),
                  &remote_addr_len) == -1) {
    LOG(ERROR) << "Failed to call 'getpeername'";
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
    LOG(ERROR) << "Failed to 'setsockopt' TCP_CONGESTION --- is CCA '" << cca
               << "' loaded?";
    return false;
  }
  std::array<char, 32> retrieved_cca{};
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(fd, SOL_TCP, TCP_CONGESTION, retrieved_cca.data(),
                 &retrieved_cca_len) == -1) {
    LOG(ERROR) << "Failed to 'getsockopt' TCP_CONGESTION";
    return false;
  }
  if (strcmp(retrieved_cca.data(), cca) != 0) {
    LOG(ERROR) << "Failed to set CCA to " << cca
               << "! Actual CCA is: " << retrieved_cca.data();
    return false;
  }
  return true;
}

// Verify that an addr is IPv4.
int check_family(const struct sockaddr *addr) {
  if (addr != nullptr && addr->sa_family != AF_INET) {
    LOG(WARNING) << "Got non-AF_INET sa_family=" << addr->sa_family;
    if (addr->sa_family == AF_INET6) {
      LOG(WARNING) << "(continued) got AF_INET6";
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
  VLOG(1) << "Found flow FD=" << fd << ": "
          << ipv4_to_string(flow.remote_addr).c_str() << ":" << flow.remote_port
          << "->" << ipv4_to_string(flow.local_addr).c_str() << ":"
          << flow.local_port;
  // Ignore flows that are not in the monitor port range.
  if (flow.remote_port < monitor_port_start ||
      flow.remote_port > monitor_port_end) {
    VLOG(1) << "Ignoring flow on remote port " << flow.remote_port
            << ", not in monitor port range: [" << monitor_port_start << ", "
            << monitor_port_end << "]";
    return;
  }
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  fd_to_flow[fd] = flow;
  rm_flow_key const key = {flow.local_addr, flow.remote_addr, flow.local_port,
                           flow.remote_port};
  flow_to_fd[key] = fd;
  addr_to_fds[flow.remote_addr].push_back(fd);
  // Change the CCA to BPF_CUBIC.
  if (!set_cca(fd, RM_BPF_CUBIC)) {
    LOG(ERROR) << "Failed to set CCA for FD=" << fd
               << ", removing from tracking";
    fd_to_flow.erase(fd);
    flow_to_fd.erase(key);
    auto &vec = addr_to_fds[flow.remote_addr];
    vec.erase(std::remove(vec.begin(), vec.end(), fd), vec.end());
    if (vec.empty())
      addr_to_fds.erase(flow.remote_addr);
    return;
  }

  // Create an entry in flow_to_last_data_time_ns for this flow so that the
  // kprobe program knows to start tracking this flow.
  int const err = bpf_map_update_elem(flow_to_last_data_time_fd,
                                      &fd_to_flow[fd], &zero, BPF_ANY);
  if (err != 0) {
    LOG(ERROR) << "Failed to create entry in flow_to_last_data_time_fd for FD="
               << fd << ", err=" << err << " (" << strerror(-err)
               << "). Removing flow from tracking.";
    // Clean up: remove from fd_to_flow, flow_to_fd, and addr_to_fds since
    // registration failed
    fd_to_flow.erase(fd);
    flow_to_fd.erase(key);
    auto &vec2 = addr_to_fds[flow.remote_addr];
    vec2.erase(std::remove(vec2.begin(), vec2.end(), fd), vec2.end());
    if (vec2.empty())
      addr_to_fds.erase(flow.remote_addr);
    return;
  }

  // All flows start paused and are added to scheduler queues.
  // In port mode, initial grants are based on port numbers (in handle_send),
  // but flows are still tracked in queues for when normal scheduling takes over
  // after the flash grants complete.
  paused_fds_queue.enqueue(fd);
  pause_flow(fd);
  VLOG(1) << "Flow FD=" << fd << " registered and added to paused queue";
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
    VLOG(1) << "Flow FD=" << fd << " is already active, doing nothing";
    return true;
  }

  if (static_cast<int>(active_fds_queue.size()) >= max_active_flows) {
    VLOG(1)
        << "Flow FD=" << fd
        << " is paused and there is no space to activate it, so doing nothing.";
    return true;
  }

  // The flow is paused but we can activate it.
  VLOG(1) << "Flow FD=" << fd
          << " is not active, but there is space to activate it, so activating "
             "it now";
  if (!paused_fds_queue.find_and_delete(fd)) {
    LOG(ERROR) << "Could not find and delete flow FD=" << fd
               << " from paused queue, this is a bug!";
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
    LOG(ERROR) << "Failed to activate flow FD=" << fd;
    return false;
  }
  VLOG(1) << "Activated flow FD=" << fd;

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
      LOG(ERROR) << "Should have cancelled 1 timer. If you are seeing this, it "
                    "is possible that the first flow to be tracked was started "
                    "before the scheduler thread could launch.";
      return false;
    }
    // The above call to timer.expires_from_now() will have cancelled any
    // other pending async_wait(), so we are safe to call async_wait() again.
    timer.async_wait(&timer_callback);
    VLOG(1) << "Exiting slow check mode.";
  }
  return true;
}

// Handle send in normal mode - uses existing queue-based scheduling logic.
// Uses lock only for queue operations, not for BPF map accesses.
static bool handle_send_normal_mode(int sockfd, const void *buf, size_t len) {
  // Update the flow_to_keepalive map to indicate that this flow has pending
  // data. BPF map has its own locking.
  if (!set_keepalive(sockfd)) {
    LOG(ERROR) << "Failed to update keepalive for FD=" << sockfd;
    return false;
  }

  // If we are doing byte-based scheduling, then track the size of this request.
  // BPF map operations don't need our lock.
  if (scheduling_mode == "byte") {
    if (len != 3 * sizeof(int)) {
      LOG(FATAL) << "FD=" << sockfd << " burst request size is " << len
                 << " bytes, expected " << 3 * sizeof(int) << " bytes (3 ints)";
      std::exit(1);
    }
    // trunk-ignore(clang-tidy/google-readability-casting)
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
    int *buf_int = (int *)buf;
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-bounds-pointer-arithmetic)
    int const burst_number = buf_int[0];
    int const bytes = buf_int[1];
    VLOG(1) << "FD=" << sockfd << " burst_number=" << burst_number << " has "
            << bytes << " bytes pending";

    struct rm_flow flow;
    {
      std::shared_lock<std::shared_mutex> lock(lock_scheduler);
      auto flow_it = fd_to_flow.find(sockfd);
      if (flow_it == fd_to_flow.end()) {
        LOG(ERROR) << "Flow FD=" << sockfd << " not found in fd_to_flow";
        return false;
      }
      flow = flow_it->second;
    }

    struct rm_grant_info grant_info {};
    int err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
    if (err != 0) {
      VLOG(1) << "Could not find existing grant info for flow FD=" << sockfd
              << ", creating new grant info";
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes = 0;
      grant_info.rwnd_end_seq = 0;
      grant_info.grant_end_seq = 0;
      grant_info.grant_done = true;
      grant_info.grant_end_buffer_bytes = grant_end_buffer_bytes;
      grant_info.bpf_experienced_error = false;
      grant_info.pregranted_bytes = 0;
    } else {
      check_bpf_error(grant_info, "register_fd_for_monitoring_single_burst");
    }
    grant_info.ungranted_bytes += bytes;
    err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
    if (err != 0) {
      LOG(ERROR) << "Could not update ungranted bytes for flow FD=" << sockfd
                 << ", err=" << err << " (" << strerror(-err) << ")";
      return false;
    }
  }

  // Use existing queue-based scheduling logic (requires lock)
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);
  if (!schedule_on_new_request(sockfd)) {
    LOG(ERROR) << "Failed to schedule flow for FD=" << sockfd;
    return false;
  }

  return true;
}

// Handle send in single mode - one send() per host represents all flows from
// that host. This mode is for IBG's single_request feature where the receiver
// sends one burst request to each sender host, and that single send() should
// update state for all flows from that host. Two policies:
// - "normal": Perform scheduling on every send() call
// - "pregrant": Perform scheduling only on first burst; subsequent bursts use
// pregrants
static bool handle_send_single_mode(int sockfd, const void *buf, size_t len) {
  // Quick check: skip unmonitored FDs (e.g., control connections) before
  // parsing the burst message. Uses a shared lock to avoid blocking other
  // readers.
  {
    std::shared_lock<std::shared_mutex> slock(lock_scheduler);
    if (fd_to_flow.find(sockfd) == fd_to_flow.end()) {
      return false;
    }
  }

  // Parse burst_number and bytes from the burst request message before
  // acquiring any lock. Format is: [burst_number, bytes, wait_us, padding]
  if (len != 4 * sizeof(int)) {
    LOG(FATAL) << "FD=" << sockfd << " single mode burst request size is "
               << len << " bytes, expected " << 4 * sizeof(int)
               << " bytes (4 ints)";
    std::exit(1);
  }
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-reinterpret-cast)
  const int *buf_int = reinterpret_cast<const int *>(buf);
  int const burst_number = buf_int[0];
  int const bytes = buf_int[1];

  // Acquire lock to access fd_to_flow and global state
  std::unique_lock<std::shared_mutex> lock(lock_scheduler);

  // Look up remote_addr from fd_to_flow (avoids getsockname/getpeername
  // syscalls that get_flow() would do).
  auto flow_it = fd_to_flow.find(sockfd);
  if (flow_it == fd_to_flow.end()) {
    LOG(ERROR) << "Could not get flow for FD=" << sockfd;
    return false;
  }
  uint32_t const remote_addr = flow_it->second.remote_addr;

  VLOG(1) << "Single mode - send() called for FD=" << sockfd << " from host "
          << ipv4_to_string(remote_addr).c_str()
          << ", received_burst_number=" << burst_number << ", bytes=" << bytes
          << ", current_burst_number=" << current_burst_number
          << ", policy=" << single_request_policy.c_str();

  // Update burst tracking BEFORE scheduling decisions
  // This ensures should_schedule is calculated based on the current burst
  if (burst_number >= 0 && burst_number > current_burst_number) {
    // New burst detected
    VLOG(1) << "Starting burst " << burst_number << " (was "
            << current_burst_number
            << "), reset burst_flows_remaining=" << fd_to_flow.size();
    current_burst_number = burst_number;
    pregrant_done = false;
    burst_flows_remaining = fd_to_flow.size();
  } else if (burst_number >= 0 && burst_number < current_burst_number) {
    LOG(WARNING) << "Received burst_number=" << burst_number
                 << " but current_burst_number=" << current_burst_number;
  }

  // Initialize burst tracking on first burst
  if (current_burst_number == 0 && burst_flows_remaining == 0) {
    // Count total flows to initialize burst_flows_remaining
    burst_flows_remaining = fd_to_flow.size();
    pregrant_done = false;
    VLOG(1) << "First burst - initialized burst_flows_remaining="
            << burst_flows_remaining;
  }

  // Determine whether to perform scheduling
  bool should_schedule = false;
  if (single_request_policy == "normal") {
    // Normal policy: always schedule
    should_schedule = true;
  } else if (single_request_policy == "pregrant") {
    // Pregrant policy: only schedule on first burst (burst_number 0)
    should_schedule = (current_burst_number == 0);
    VLOG(1) << "Pregrant policy - current_burst_number=" << current_burst_number
            << ", should_schedule=" << should_schedule;
  }

  // Find all flows from the same remote host using the per-host index.
  int flows_updated = 0;
  auto ait = addr_to_fds.find(remote_addr);

  if (!should_schedule) {
    // Defer BPF map updates — cache burst info per host instead of iterating
    // all flows. Each flow's BPF maps (keepalive + ungranted_bytes) will be
    // updated lazily when the flow is activated via handle_grant_done() or
    // try_activate_one().
    pending_burst_info[remote_addr] = {bytes, current_burst_number};
    flows_updated =
        (ait != addr_to_fds.end()) ? static_cast<int>(ait->second.size()) : 0;
    // Clear between_bursts while still holding the lock, then release.
    between_bursts = false;
    lock.unlock();
    VLOG(1) << "Single mode (pregrant) - deferred " << bytes << " bytes for "
            << flows_updated << " flows from "
            << ipv4_to_string(remote_addr).c_str()
            << ", burst=" << current_burst_number;
    return true;
  } else if (ait != addr_to_fds.end()) {
    // should_schedule=true (first burst or normal policy): eagerly update all
    // flows' BPF maps and perform scheduling.
    for (int const fd : ait->second) {
      auto fit = fd_to_flow.find(fd);
      if (fit == fd_to_flow.end()) {
        continue;
      }
      struct rm_flow const &flow_iter = fit->second;
      flows_updated++;

      VLOG(1) << "Single mode - updating flow FD=" << fd
              << " (remote_port=" << flow_iter.remote_port << ") from host "
              << ipv4_to_string(remote_addr).c_str();

      // Set keepalive for this flow (BPF map - has internal locking)
      if (!set_keepalive(fd)) {
        LOG(ERROR) << "Could not set keepalive for flow FD=" << fd;
        // Continue with other flows rather than failing completely
      }

      // Update ungranted bytes in BPF map (BPF map - has internal locking)
      int err = 0;
      struct rm_grant_info grant_info {};
      err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow_iter, &grant_info);
      if (err != 0) {
        LOG(WARNING) << "Could not find grant info for flow FD=" << fd
                     << ", creating new entry";
        grant_info = {};
        grant_info.override_rwnd_bytes = 0xFFFFFFFF;
        grant_info.new_grant_bytes = 0;
        grant_info.rwnd_end_seq = 0;
        grant_info.grant_end_seq = 0;
        grant_info.grant_done = true;
        grant_info.grant_end_buffer_bytes = grant_end_buffer_bytes;
        grant_info.bpf_experienced_error = false;
        grant_info.pregranted_bytes = 0;
      } else {
        check_bpf_error(grant_info, "handle_send_single_mode");
      }

      // Add the burst size to ungranted bytes
      grant_info.ungranted_bytes += bytes;
      err = bpf_map_update_elem(flow_to_rwnd_fd, &flow_iter, &grant_info,
                                BPF_ANY);
      if (err != 0) {
        LOG(ERROR) << "Could not update grant info for flow FD=" << fd
                   << ", err=" << err << " (" << strerror(-err) << ")";
        // Continue with other flows
      }

      VLOG(1) << "Single mode - flow FD=" << fd << " now has "
              << grant_info.ungranted_bytes << " ungranted bytes";

      // Call schedule_on_new_request for this flow
      // This will activate the flow if there's capacity, otherwise leave it
      // paused
      if (!schedule_on_new_request(fd)) {
        LOG(WARNING) << "schedule_on_new_request failed for flow FD=" << fd;
        // Continue with other flows
      }

      // Trigger an ACK for flows that are activated, except for the calling
      // flow (sockfd) which will carry the grant in the burst request itself.
      if (fd != sockfd) {
        // NOTE: These ACKs will probably go out slowly due to performance
        // issues with sending packets on many flows, so the first burst will
        // have bad performance. This is okay. We care about steady state
        // (later bursts).
        trigger_ack(fd);
      } else {
        VLOG(1) << "Single mode - skipping ACK trigger for calling flow FD="
                << fd << " (grant will be in burst request)";
      }
    }
  }

  VLOG(1) << "Single mode - updated " << flows_updated << " flows from host "
          << ipv4_to_string(remote_addr).c_str();

  if (flows_updated == 0) {
    LOG(WARNING) << "Single mode - no flows found for host "
                 << ipv4_to_string(remote_addr).c_str()
                 << " (only found FD=" << sockfd << ")";
  }

  // Clear between_bursts now that a new burst request has been fully processed.
  // This must happen after all send processing so handle_grant_done can detect
  // spurious calls during the quiescent period.
  between_bursts = false;

  return true;
}

// Handle send in port mode - makes decisions based on port numbers.
// Provides initial "flash" grants, then queue-based scheduling takes over.
// Uses shared lock for reading flow info, then unique lock for queue
// manipulation.
static bool handle_send_port_mode(int sockfd, const void *buf, size_t len) {
  // First, extract needed information with a shared lock
  uint16_t remote_port;
  struct rm_flow flow;
  {
    std::shared_lock<std::shared_mutex> lock(lock_scheduler);

    auto flow_it = fd_to_flow.find(sockfd);
    if (flow_it == fd_to_flow.end()) {
      LOG(ERROR) << "Flow FD=" << sockfd << " not found in fd_to_flow";
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
    LOG(ERROR) << "Could not update flow_to_keepalive for FD=" << sockfd
               << ", err=" << err << " (" << strerror(-err) << ")";
    return false;
  }
  VLOG(1) << "Updated flow_to_keepalive for FD=" << sockfd;

  // If we are doing byte-based scheduling, then track the size of this request.
  if (scheduling_mode == "byte") {
    // Parse burst request to get number of bytes in the burst
    // Format is: [burst_number, bytes, wait_us, padding]
    if (len != 4 * sizeof(int)) {
      LOG(FATAL) << "FD=" << sockfd << " port mode burst request size is "
                 << len << " bytes, expected " << 4 * sizeof(int)
                 << " bytes (4 ints)";
      std::exit(1);
    }
    // trunk-ignore(clang-tidy/google-readability-casting)
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
    int *buf_int = (int *)buf;
    // trunk-ignore(clang-tidy/cppcoreguidelines-pro-bounds-pointer-arithmetic)
    int const burst_number = buf_int[0];
    int const bytes = buf_int[1];
    VLOG(1) << "FD=" << sockfd << " burst_number=" << burst_number << " has "
            << bytes << " bytes pending";

    struct rm_grant_info grant_info {};
    err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
    if (err != 0) {
      VLOG(1) << "Could not find existing grant info for flow FD=" << sockfd
              << ", creating new grant info";
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes = 0;
      grant_info.rwnd_end_seq = 0;
      grant_info.grant_end_seq = 0;
      grant_info.grant_done = true;
      grant_info.grant_end_buffer_bytes = grant_end_buffer_bytes;
      grant_info.bpf_experienced_error = false;
      grant_info.pregranted_bytes = 0;
    } else {
      check_bpf_error(grant_info, "handle_send_port_mode_ungranted");
    }
    grant_info.ungranted_bytes += bytes;
    err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
    if (err != 0) {
      LOG(ERROR) << "Could not update ungranted bytes for flow FD=" << sockfd
                 << ", err=" << err << " (" << strerror(-err) << ")";
      return false;
    }
  }

  // Port mode: make scheduling decision based on remote port number.
  // This provides the initial "flash" grant. After this grant completes,
  // handle_grant_done will transition the flow to normal queue-based
  // scheduling. NOTE: We do NOT touch queue data structures here. The queues
  // will show all flows as paused during flash grants, which is fine - normal
  // scheduling takes over when flash grants complete.
  uint16_t const port_range_end = monitor_port_start + max_active_flows;

  if (remote_port >= monitor_port_start && remote_port < port_range_end) {
    // This flow gets a flash grant - set grant in BPF maps only, don't touch
    // queues
    VLOG(1) << "Port mode - giving flash grant to flow FD=" << sockfd
            << " with remote_port=" << remote_port << " (in range ["
            << monitor_port_start << ", " << port_range_end << "))";

    // Set grant for byte-based scheduling or remove RWND clamp for time-based
    if (scheduling_mode == "byte") {
      struct rm_grant_info grant_info {};
      err = bpf_map_lookup_elem(flow_to_rwnd_fd, &flow, &grant_info);
      if (err != 0) {
        LOG(ERROR) << "Could not find existing grant for flow FD=" << sockfd;
        return false;
      }
      check_bpf_error(grant_info, "handle_send_port_mode");
      if (grant_info.ungranted_bytes <= 0) {
        VLOG(1) << "Cannot activate flow FD=" << sockfd
                << " because it has no ungranted bytes";
        return true;
      }
      grant_info.override_rwnd_bytes = 0xFFFFFFFF;
      grant_info.new_grant_bytes += epoch_bytes;
      grant_info.grant_done = false;
      grant_info.is_pregrant = false;
      err = bpf_map_update_elem(flow_to_rwnd_fd, &flow, &grant_info, BPF_ANY);
      if (err != 0) {
        LOG(ERROR) << "Could not set grant for flow FD=" << sockfd
                   << ", err=" << err << " (" << strerror(-err) << ")";
        return false;
      }
      VLOG(1) << "Gave flash grant of " << epoch_bytes
              << " bytes to flow FD=" << sockfd << " (queues not updated)";
    } else if (scheduling_mode == "time") {
      err = bpf_map_delete_elem(flow_to_rwnd_fd, &flow);
      if (err != 0) {
        LOG(WARNING) << "Could not delete RWND clamp for flow FD=" << sockfd
                     << ", err=" << err << " (" << strerror(-err) << ")";
      }
      VLOG(1) << "Gave flash grant to flow FD=" << sockfd
              << " in time-based mode (queues not updated)";
    }
    // Note: Don't trigger ACK since grant will be in the outgoing request
  } else {
    // This flow doesn't get a flash grant - stays paused in queue
    // It will be activated by normal queue-based scheduling later
    VLOG(1) << "Port mode - flow FD=" << sockfd
            << " with remote_port=" << remote_port << " (not in range ["
            << monitor_port_start << ", " << port_range_end
            << ")) stays paused, will be activated by normal scheduling";
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
    LOG(ERROR) << "Failed to query dlsym for 'accept': " << dlerror();
    return -1;
  }
  int const fd = real_accept(sockfd, addr, addrlen);
  if (fd == -1) {
    LOG(ERROR) << "Real 'accept' failed";
    return fd;
  }
  if (noop_mode) {
    return fd;
  }
  if (check_family(addr) != 0) {
    return fd;
  }
  if (!setup_done) {
    LOG(ERROR) << "Cannot handle 'accept', setup not done, returning FD=" << fd;
    return fd;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    VLOG(1) << "libratemon_interp not running, 'accept' returning FD=" << fd;
    return fd;
  }

  register_fd_for_monitoring(fd);
  {
    std::shared_lock<std::shared_mutex> lock(lock_scheduler);
    if (fd_to_flow.find(fd) == fd_to_flow.end()) {
      LOG(WARNING)
          << "'accept' FD=" << fd
          << " was NOT registered for monitoring (registration failed)";
    }
  }
  VLOG(1) << "Successful 'accept' for FD=" << sockfd << ", got FD=" << fd;
  return fd;
}

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
  // trunk-ignore(clang-tidy/google-readability-casting)
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
  static auto real_connect = (int (*)(int, const struct sockaddr *, socklen_t))(
      dlsym(RTLD_NEXT, "connect"));
  if (real_connect == nullptr) {
    LOG(ERROR) << "Failed to query dlsym for 'connect': " << dlerror();
    return -1;
  }
  int const fd = real_connect(sockfd, addr, addrlen);
  if (fd == -1) {
    LOG(ERROR) << "Real 'connect' failed";
    return fd;
  }
  if (noop_mode) {
    return fd;
  }
  if (check_family(addr) != 0) {
    return fd;
  }
  if (!setup_done) {
    LOG(ERROR) << "Cannot handle 'connect', setup not done, returning FD="
               << fd;
    return fd;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    VLOG(1) << "libratemon_interp not running, 'connect' returning FD=" << fd;
    return fd;
  }

  register_fd_for_monitoring(fd);
  {
    std::shared_lock<std::shared_mutex> lock(lock_scheduler);
    if (fd_to_flow.find(fd) == fd_to_flow.end()) {
      LOG(WARNING)
          << "'connect' FD=" << fd
          << " was NOT registered for monitoring (registration failed)";
    }
  }
  VLOG(1) << "Successful 'connect' for FD=" << sockfd << ", got FD=" << fd;
  return fd;
}

ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
  static auto real_send = (
      // trunk-ignore(clang-tidy/google-readability-casting)
      // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
      (ssize_t(*)(int, const void *, size_t, int))dlsym(RTLD_NEXT, "send"));
  if (real_send == nullptr) {
    LOG(ERROR) << "Failed to query dlsym for 'send': " << dlerror();
    return -1;
  }
  if (noop_mode) {
    return real_send(sockfd, buf, len, flags);
  }
  // Do any ratemon-specific handling before calling the real send().
  // handle_send functions do their own fd_to_flow check under lock, so we
  // skip the separate is_monitored shared lock to avoid double-locking.
  if (setup_done && run) {
    if (!handle_send(sockfd, buf, len)) {
      LOG(ERROR) << "Failed to handle 'send' for FD=" << sockfd;
    }
  }
  ssize_t const ret = real_send(sockfd, buf, len, flags);
  if (ret == -1) {
    LOG(ERROR) << "Real 'send' failed for FD=" << sockfd;
    return ret;
  }
  VLOG(1) << "Successful 'send' for FD=" << sockfd << ", sent " << ret
          << " bytes";
  return ret;
}

// Get around C++ function name mangling.
extern "C" {
int close(int sockfd) {
  // trunk-ignore(clang-tidy/cppcoreguidelines-pro-type-cstyle-cast)
  static auto real_close = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == nullptr) {
    LOG(ERROR) << "Failed to query dlsym for 'close': " << dlerror();
    return -1;
  }
  int const ret = real_close(sockfd);
  if (ret == -1) {
    LOG(ERROR) << "Real 'close' failed";
  } else {
    VLOG(1) << "Successful 'close' for FD=" << sockfd;
  }
  if (noop_mode) {
    return ret;
  }
  if (!setup_done) {
    LOG(ERROR) << "Cannot handle 'close', setup not done, returning FD="
               << sockfd;
    return ret;
  }
  // If we have been signalled to quit, then do nothing more.
  if (!run) {
    VLOG(1) << "libratemon_interp not running, returning from 'close' for FD="
            << sockfd;
    return ret;
  }

  // Quick check if this flow is monitored before acquiring the lock.
  // This avoids overhead for non-monitored connections.
  bool is_monitored = false;
  {
    std::shared_lock<std::shared_mutex> read_lock(lock_scheduler);
    is_monitored = (fd_to_flow.find(sockfd) != fd_to_flow.end());
  }

  if (!is_monitored) {
    VLOG(1) << "Ignoring 'close' for FD=" << sockfd << ", not in fd_to_flow";
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
    VLOG(1) << "Removed FD=" << sockfd << " from fd_to_flow (" << removed
            << " elements removed)";
    removed = flow_to_fd.erase(key);
    VLOG(1) << "Removed FD=" << sockfd << " from flow_to_fd (" << removed
            << " elements removed)";
    // Remove from per-host index.
    auto ait = addr_to_fds.find(flow.remote_addr);
    if (ait != addr_to_fds.end()) {
      auto &vec = ait->second;
      vec.erase(std::remove(vec.begin(), vec.end(), sockfd), vec.end());
      if (vec.empty())
        addr_to_fds.erase(ait);
    }
    // Remove deferred burst tracking for this FD.
    fd_last_applied_burst.erase(sockfd);

    // If this is the last flow that we know about and we close it, then assume
    // that we are no longer needed and kill the scheduler thread.
    if (fd_to_flow.empty()) {
      VLOG(1) << "No more flows remaining, stopping libratemon_interp and its "
                 "scheduler thread.";
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

    // Remove the flow from the active_fds_queue if it was active, and try to
    // activate a replacement flow so that scheduling does not stall.
    bool was_active = false;
    int queue_size = static_cast<int>(active_fds_queue.size());
    for (int i = 0; i < queue_size; ++i) {
      auto front = active_fds_queue.front();
      active_fds_queue.pop();
      if (front.first == sockfd) {
        was_active = true;
        VLOG(1) << "Removed closed FD=" << sockfd << " from active_fds_queue";
        // Don't push it back.
      } else {
        active_fds_queue.push(front);
      }
    }
    // Also remove from paused queue if present.
    paused_fds_queue.find_and_delete(sockfd);

    if (was_active && !paused_fds_queue.empty()) {
      // Try to activate a replacement flow so we don't waste a slot.
      int const num_activated = try_activate_one();
      VLOG(1) << "Closed active FD=" << sockfd << ", activated "
              << num_activated << " replacement flow(s)";
    }
  }
  // Note: If fd_to_flow doesn't contain sockfd at this point, another thread
  // may have removed it between our check and acquiring the lock. This is fine.
  // lock is automatically released when it goes out of scope
  return ret;
}
}

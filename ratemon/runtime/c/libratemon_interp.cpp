#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <dlfcn.h>
#include <linux/inet_diag.h>
#include <netinet/in.h> // structure for storing address information
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h> // for socket APIs
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cmath>
#include <experimental/random>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ratemon.h"

// Protects writes only to max_active_flows, epoch_us, idle_timeout_ns,
// monitor_port_start, monitor_port_end, flow_to_rwnd_fd, flow_to_win_scale_fd,
// flow_to_last_data_time_fd, flow_to_keepalive_fd, oldact, and setup. Reads are
// unprotected.
std::mutex lock_setup;
// Whether setup has been performed.
bool setup_done = false;
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
// Runs async timers for scheduling
boost::asio::io_service io;
// Periodically performs scheduling using timer_callback().
boost::asio::deadline_timer timer(io);
// Manages the io_service.
boost::thread scheduler_thread;
// Protects writes and reads to active_fds_queue, paused_fds_queue, and
// fd_to_flow.
std::mutex lock_scheduler;
// FDs for flows thare are currently active.
std::queue<std::pair<int, boost::posix_time::ptime>> active_fds_queue;
// FDs for flows that are currently paused (RWND = 0 B);
std::queue<int> paused_fds_queue;
// Maps file descriptor to rm_flow struct.
std::unordered_map<int, struct rm_flow> fd_to_flow;
// The next four are scheduled RWND tuning parameters. See ratemon.h for
// parameter documentation.
unsigned int max_active_flows = 5;
unsigned int epoch_us = 10000;
long idle_timeout_us = 0;
unsigned long idle_timeout_ns = 0;
unsigned short monitor_port_start = 9000;
unsigned short monitor_port_end = 9999;

// Used to set entries in flow_to_rwnd.
int zero = 0;
boost::posix_time::seconds one_sec = boost::posix_time::seconds(1);
// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length = (socklen_t)sizeof(placeholder_cc_info);

// Trigger a pure ACK packet to be send on this FD by calling getsockopt() with
// TCP_CC_INFO. This only works if the flow is using the CCA BPF_CUBIC.
inline void trigger_ack(int fd) {
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&placeholder_cc_info,
             &placeholder_cc_info_length);
}

// Jitter the provided value by +/- 12.5%. Returns just the jitter.
inline int jitter(int v) {
  return std::experimental::randint(0, (int)std::roundl(v * 0.25)) -
         (int)std::roundl(v * 0.125);
}

inline void activate_flow(int fd) {
  bpf_map_delete_elem(flow_to_rwnd_fd, &fd_to_flow[fd]);
  trigger_ack(fd);
  RM_PRINTF("INFO: activated FD=%d\n", fd);
}

inline void pause_flow(int fd) {
  // Pausing a flow means retting its RWND to 0 B.
  bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[fd], &zero, BPF_ANY);
  trigger_ack(fd);
  RM_PRINTF("INFO: paused flow FD=%d\n", fd);
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
  RM_PRINTF("INFO: in timer_callback\n");

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
    RM_PRINTF("INFO: program signalled to exit\n");
    return;
  }
  // If setup has not been performed yet, then we cannot perform scheduling.
  // Otherwise, revert to slow check mode.
  if (!setup_done) {
    RM_PRINTF("INFO: not set up\n");
    if (timer.expires_from_now(one_sec)) {
      RM_PRINTF("ERROR: timer unexpectedly cancelled\n");
    }
    timer.async_wait(&timer_callback);
    return;
  }
  // Check that relevant parameters have been set. Otherwise, revert to slow
  // check mode.
  if (!max_active_flows || !epoch_us || !flow_to_rwnd_fd ||
      !flow_to_last_data_time_fd || !flow_to_keepalive_fd) {
    RM_PRINTF(
        "ERROR: cannot continue, invalid max_active_flows=%u, epoch_us=%u, "
        "flow_to_rwnd_fd=%d, flow_to_last_data_time_fd=%d, or "
        "flow_to_keepalive_fd=%d\n",
        max_active_flows, epoch_us, flow_to_rwnd_fd, flow_to_last_data_time_fd,
        flow_to_keepalive_fd);
    if (timer.expires_from_now(one_sec)) {
      RM_PRINTF("ERROR: timer unexpectedly cancelled\n");
    }
    timer.async_wait(&timer_callback);
    return;
  }

  // It is now safe to perform scheduling.
  lock_scheduler.lock();
  RM_PRINTF("INFO: performing scheduling. active=%lu, paused=%lu\n",
            active_fds_queue.size(), paused_fds_queue.size());

  // Temporary variable for storing the front of active_fds_queue.
  std::pair<int, boost::posix_time::ptime> a;
  // Temporary variable for storing the front of paused_fds_queue.
  int p;
  // Size of active_fds_queue.
  unsigned long s;
  // Vector of active flows that we plan to pause.
  std::vector<int> to_pause;
  // Current kernel time (since boot).
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  unsigned long ktime_now_ns = ts.tv_sec * 1000000000ull + ts.tv_nsec;
  // For measuring idle time.
  unsigned long last_data_time_ns, idle_ns;
  // Current time (absolute).
  boost::posix_time::ptime now =
      boost::posix_time::microsec_clock::local_time();
  // New epoch time.
  boost::posix_time::ptime now_plus_epoch =
      now + boost::posix_time::microseconds(epoch_us);

  // Typically, active_fds_queue will be small and paused_fds_queue will be
  // large. Therefore, it is alright for us to iterate through the entire
  // active_fds_queue (multiple times), but we must iterate through as few
  // elements of paused_fds_queue as possible.

  // 1) Perform a status check on all active flows. It is alright to iterate
  // through all of active_fds_queue.
  s = active_fds_queue.size();
  for (unsigned long i = 0; i < s; ++i) {
    a = active_fds_queue.front();
    active_fds_queue.pop();
    // 1.1) If this flow has been closed, remove it.
    if (!fd_to_flow.contains(a.first))
      continue;
    // 1.2) If idle timeout mode is enabled, then check if this flow is
    // past its idle timeout. Skip this check if there are no paused
    // flows.
    if (idle_timeout_ns > 0 && !paused_fds_queue.empty()) {
      // Look up this flow's last active time.
      if (!bpf_map_lookup_elem(flow_to_last_data_time_fd, &fd_to_flow[a.first],
                               &last_data_time_ns)) {
        // If last_data_time_ns is 0, then this flow has not yet been tracked.
        if (last_data_time_ns) {
          if (last_data_time_ns > ktime_now_ns) {
            // This could be fine...perhaps a packet arrived since we captured
            // the current time above?
            RM_PRINTF(
                "WARNING: FD=%d last data time (%lu ns) is more recent that "
                "current time (%lu ns) by %lu ns\n",
                a.first, last_data_time_ns, ktime_now_ns,
                last_data_time_ns - ktime_now_ns);
          } else {
            idle_ns = ktime_now_ns - last_data_time_ns;
            RM_PRINTF("INFO: FD=%d now: %lu ns, last data time: %lu ns\n",
                      a.first, ktime_now_ns, last_data_time_ns);
            RM_PRINTF(
                "INFO: FD=%d idle has been idle for %lu ns. timeout is %lu "
                "ns\n",
                a.first, idle_ns, idle_timeout_ns);
            // If the flow has been idle for longer than the idle timeout, then
            // pause it. We pause the flow *before* activating a replacement
            // flow because it is by definition not sending data, so we do not
            // risk causing a drop in utilization by pausing it immediately.
            if (idle_ns >= idle_timeout_ns) {
              RM_PRINTF("INFO: Pausing FD=%d due to idle timeout\n", a.first);
              // Remove the flow from flow_to_keepalive, signalling that it no
              // longer has pending demand.
              bpf_map_delete_elem(flow_to_keepalive_fd, &a.first);
              paused_fds_queue.push(a.first);
              pause_flow(a.first);
              continue;
            }
          }
        }
      }
    }
    // 1.3) If the flow has been active for longer than its epoch, then plan to
    // pause it.
    if (now > a.second) {
      if (paused_fds_queue.empty()) {
        // If there are no paused flows, then immediately reactivate this flow.
        // Randomly jitter the epoch time by +/- 12.5%.
        active_fds_queue.push(
            {a.first, now_plus_epoch +
                          boost::posix_time::microseconds(jitter(epoch_us))});
        RM_PRINTF("INFO: reactivated FD=%d\n", a.first);
        continue;
      } else {
        // Plan to pause this flow.
        to_pause.push_back(a.first);
      }
    }
    // Put the flow back in the active queue. For this to occur, we know the
    // flow is not idle, and it is either not yet at its epoch time or it has
    // passed its epoch time and it will be extracted and paused later.
    active_fds_queue.push(a);
  }

  // 2) Activate flows. Now we can calculate how many flows to activate to reach
  // full capacity. This value is the existing free capacity plus the number of
  // flows we intend to pause. The important part here is that we only look at
  // as many entries in paused_fds_queue as needed.
  unsigned long num_to_activate =
      max_active_flows - active_fds_queue.size() + to_pause.size();
  int dummy;
  for (unsigned long i = 0; i < num_to_activate; ++i) {
    // Loop until we find a paused flow that is valid (not closed).
    unsigned long num_paused = paused_fds_queue.size();
    for (unsigned long j = 0; j < num_paused; ++j) {
      p = paused_fds_queue.front();
      paused_fds_queue.pop();
      // If this flow has been closed, then skip it.
      if (!fd_to_flow.contains(p))
        continue;
      // If this flow is not in the flow_to_keepalive map (bpf_map_lookup_elem()
      // returns negative error code when the flow is not found), then it has no
      // pending data and should be skipped.
      if (bpf_map_lookup_elem(flow_to_keepalive_fd, &p, &dummy)) {
        // RM_PRINTF("INFO: skipping activating FD=%d, no pending data\n", p);
        paused_fds_queue.push(p);
        continue;
      }
      // Randomly jitter the epoch time by +/- 12.5%.
      active_fds_queue.push(
          {p,
           now_plus_epoch + boost::posix_time::microseconds(jitter(epoch_us))});
      activate_flow(p);
      break;
    }
  }

  // 3) Pause flows. We need to recalculate the number of flows to pause because
  // we may not have been able to activate as many flows as planned. Recall that
  // it is alright to iterate through all of active_fds_queue.
  unsigned long num_to_pause = (unsigned long)std::max(
      0L, (long)active_fds_queue.size() - (long)max_active_flows);
#ifdef RM_VERBOSE
  assert(num_to_pause <= to_pause.size());
#endif
  // For each flow that we are supposed to pause, advance through
  // active_fds_queue until we find it.
  s = active_fds_queue.size();
  unsigned long j = 0;
  for (unsigned long i = 0; i < num_to_pause; ++i) {
    while (j < s) {
      a = active_fds_queue.front();
      active_fds_queue.pop();
      ++j;
      if (a.first == to_pause[i]) {
        // Pause this flow.
        paused_fds_queue.push(a.first);
        pause_flow(a.first);
        break;
      }
      // Examine the next flow in active_fds_queue.
      active_fds_queue.push(a);
    }
  }

  // 4) Check invariants.
#ifdef RM_VERBOSE
  // Cannot have more than the max number of active flows.
  assert(active_fds_queue.size() <= max_active_flows);
  // If there are no active flows, then there should also be no paused flows.
  // No, this is not strictly true anymore. If none of the flows have pending
  // data (i.e., none are in flow_to_keepalive), then they will all be paused.
  // assert(!active_fds_queue.empty() || paused_fds_queue.empty());
#endif

  // 5) Calculate when the next timer should expire.
  boost::posix_time::time_duration when;
  long next_epoch_us =
      (active_fds_queue.front().second - now).total_microseconds();
  if (active_fds_queue.empty()) {
    // If there are no flows, revert to slow check mode.
    RM_PRINTF("INFO: no flows remaining, reverting to slow check mode\n");
    when = one_sec;
  } else if (!idle_timeout_ns) {
    // If we are not using idle timeout mode...
    RM_PRINTF("INFO: no idle timeout, scheduling timer for next epoch end\n");
    when = boost::posix_time::microsec(next_epoch_us);
  } else if (idle_timeout_us < next_epoch_us) {
    // If we are using idle timeout mode...
    RM_PRINTF("INFO: scheduling timer for next idle timeout\n");
    when = boost::posix_time::microsec(idle_timeout_us);
  } else {
    RM_PRINTF("INFO: scheduling timer for next epoch end, sooner than idle "
              "timeout\n");
    when = boost::posix_time::microsec(next_epoch_us);
  }

  // 6) Start the next timer.
  if (timer.expires_from_now(when)) {
    RM_PRINTF("ERROR: timer unexpectedly cancelled\n");
  }
  timer.async_wait(&timer_callback);
  lock_scheduler.unlock();
  RM_PRINTF("INFO: sleeping until next event in %ld us\n",
            when.total_microseconds());
  return;
}

// This function is designed to be run in a thread. It is responsible for
// managing the async timers that perform scheduling. The timer events are
// executed by this thread, but they can be scheduled by other threads.
void thread_func() {
  RM_PRINTF("INFO: scheduler thread started\n");
  if (timer.expires_from_now(one_sec)) {
    RM_PRINTF("ERROR: timer unexpectedly cancelled\n");
  }

  timer.async_wait(&timer_callback);
  RM_PRINTF("INFO: scheduler thread initial sleep\n");
  // Execute the configured events, until there are no more events to execute.
  io.run();

  // Delete all flows from flow_to_rwnd and flow_to_win_scale.
  lock_scheduler.lock();
  for (const auto &p : fd_to_flow) {
    if (flow_to_rwnd_fd)
      bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);
    if (flow_to_win_scale_fd)
      bpf_map_delete_elem(flow_to_win_scale_fd, &p.second);
    if (flow_to_last_data_time_fd)
      bpf_map_delete_elem(flow_to_last_data_time_fd, &p.second);
    if (flow_to_keepalive_fd)
      bpf_map_delete_elem(flow_to_keepalive_fd, &p.second);
  }
  lock_scheduler.unlock();
  RM_PRINTF("INFO: scheduler thread ended\n");

  if (run) {
    RM_PRINTF("ERROR: scheduled thread ended before program was signalled to "
              "stop\n");
  }
}

// Catch SIGINT and trigger the scheduler thread and timer to end.
void sigint_handler(int signum) {
  switch (signum) {
  case SIGINT:
    RM_PRINTF("INFO: caught SIGINT\n");
    run = false;
    scheduler_thread.join();
    RM_PRINTF("INFO: resetting old SIGINT handler\n");
    sigaction(SIGINT, &oldact, NULL);
    break;
  default:
    RM_PRINTF("ERROR: caught signal %d\n", signum);
    break;
  }
  RM_PRINTF("INFO: re-raising signal %d\n", signum);
  raise(signum);
}

bool read_env_uint(const char *key, volatile unsigned int *dest,
                   bool allow_zero = false) {
  // Read an environment variable as an unsigned integer.
  char *val_str = getenv(key);
  if (val_str == NULL) {
    RM_PRINTF("ERROR: failed to query environment variable '%s'\n", key);
    return false;
  }
  int val_int = atoi(val_str);
  if (allow_zero and val_int < 0) {
    RM_PRINTF("ERROR: invalid value for '%s'=%d (must be > 0)\n", key, val_int);
    return false;
  } else if (!allow_zero and val_int <= 0) {
    RM_PRINTF("ERROR: invalid value for '%s'=%d (must be >= 0)\n", key,
              val_int);
    return false;
  }
  *dest = (unsigned int)val_int;
  return true;
}

// Perform setup (only once for all flows in this process), such as reading
// parameters from environment variables and looking up the BPF map
// flow_to_rwnd.
bool setup() {
  // Read environment variables with parameters.
  if (!read_env_uint(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows))
    return false;
  if (!read_env_uint(RM_EPOCH_US_KEY, &epoch_us))
    return false;
  unsigned int idle_timeout_us_;
  if (!read_env_uint(RM_IDLE_TIMEOUT_US_KEY, &idle_timeout_us_,
                     true /* allow_zero */))
    return false;
  idle_timeout_us = (long)idle_timeout_us_;
  idle_timeout_ns = (unsigned long)idle_timeout_us * 1000UL;
  unsigned int monitor_port_start_;
  if (!read_env_uint(RM_MONITOR_PORT_START_KEY, &monitor_port_start_) ||
      monitor_port_start_ >= 65536)
    return false;
  monitor_port_start = (unsigned short)monitor_port_start_;
  unsigned int monitor_port_end_;
  if (!read_env_uint(RM_MONITOR_PORT_END_KEY, &monitor_port_end_) ||
      monitor_port_end_ >= 65536)
    return false;
  monitor_port_end = (unsigned short)monitor_port_end_;

  // Look up the FD for the flow_to_rwnd map. We do not need the BPF skeleton
  // for this.
  int err = bpf_obj_get(RM_FLOW_TO_RWND_PIN_PATH);
  if (err == -1) {
    RM_PRINTF("ERROR: failed to get FD for 'flow_to_rwnd' from path '%s'\n",
              RM_FLOW_TO_RWND_PIN_PATH);
    return false;
  }
  flow_to_rwnd_fd = err;

  // Look up the FD for the flow_to_win_scale map. We do not need the BPF
  // skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_WIN_SCALE_PIN_PATH);
  if (err == -1) {
    RM_PRINTF(
        "ERROR: failed to get FD for 'flow_to_win_scale' from path '%s'\n",
        RM_FLOW_TO_WIN_SCALE_PIN_PATH);
    return false;
  }
  flow_to_win_scale_fd = err;

  // Look up the FD for the flow_to_last_data_time_ns map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH);
  if (err == -1) {
    RM_PRINTF(
        "ERROR: failed to get FD for 'flow_to_last_data_time_ns' from path "
        "'%s'\n",
        RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH);
    return false;
  }
  flow_to_last_data_time_fd = err;

  // Look up the FD for the flow_to_keepalive map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_KEEPALIVE_PIN_PATH);
  if (err == -1) {
    RM_PRINTF("ERROR: failed to get FD for 'flow_to_keepalive' from path "
              "'%s'\n",
              RM_FLOW_TO_KEEPALIVE_PIN_PATH);
    return false;
  }
  flow_to_keepalive_fd = err;

  // Catch SIGINT to end the program.
  struct sigaction action;
  action.sa_handler = sigint_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, &oldact);

  // Launch the scheduler thread.
  scheduler_thread = boost::thread(thread_func);

  RM_PRINTF("INFO: setup complete! max_active_flows=%u, epoch_us=%u, "
            "idle_timeout_ns=%lu, monitor_port_start=%u, monitor_port_end=%u\n",
            max_active_flows, epoch_us, idle_timeout_ns, monitor_port_start,
            monitor_port_end);
  return true;
}

// Fill in the four-tuple for this socket.
bool get_flow(int fd, struct rm_flow *flow) {
  // Determine the four-tuple, which we need to track because RWND tuning is
  // applied based on four-tuple.
  struct sockaddr_in local_addr;
  socklen_t local_addr_len = sizeof(local_addr);
  // Get the local IP and port.
  if (getsockname(fd, (struct sockaddr *)&local_addr, &local_addr_len) == -1) {
    RM_PRINTF("ERROR: failed to call 'getsockname'\n");
    return false;
  }
  struct sockaddr_in remote_addr;
  socklen_t remote_addr_len = sizeof(remote_addr);
  // Get the peer's (i.e., the remote) IP and port.
  if (getpeername(fd, (struct sockaddr *)&remote_addr, &remote_addr_len) ==
      -1) {
    RM_PRINTF("ERROR: failed to call 'getpeername'\n");
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
    RM_PRINTF("ERROR: failed to 'setsockopt' TCP_CONGESTION\n");
    return false;
  }
  char retrieved_cca[32];
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(fd, SOL_TCP, TCP_CONGESTION, retrieved_cca,
                 &retrieved_cca_len) == -1) {
    RM_PRINTF("ERROR: failed to 'getsockopt' TCP_CONGESTION\n");
    return false;
  }
  if (strcmp(retrieved_cca, cca)) {
    RM_PRINTF("ERROR: failed to set CCA to %s! Actual CCA is: %s\n", cca,
              retrieved_cca);
    return false;
  }
  return true;
}

// Perform initial scheduling for this flow.
void initial_scheduling(int fd) {
  // Create an entry in flow_to_last_data_time_ns for this flow so that the
  // kprobe program knows to start tracking this flow.
  bpf_map_update_elem(flow_to_last_data_time_fd, &fd_to_flow[fd], &zero,
                      BPF_ANY);
  // Should this flow be active or paused?
  if (active_fds_queue.size() < max_active_flows) {
    // Less than the max number of flows are active, so make this one active.
    boost::posix_time::ptime now =
        boost::posix_time::microsec_clock::local_time();
    active_fds_queue.push(
        {fd, now + boost::posix_time::microseconds(epoch_us) +
                 boost::posix_time::microseconds(jitter(epoch_us))});
    RM_PRINTF("INFO: allowing new flow FD=%d\n", fd);
    if (active_fds_queue.size() == 1) {
      if (timer.expires_from_now(active_fds_queue.front().second - now) != 1) {
        RM_PRINTF("ERROR: should have cancelled 1 timer\n");
      }
      timer.async_wait(&timer_callback);
      RM_PRINTF("INFO: first scheduling event\n");
    }
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds_queue.push(fd);
    pause_flow(fd);
  }
}

// For some reason, C++ function name mangling does not prevent us from
// overriding accept(), so we do not need 'extern "C"'.
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  static int (*real_accept)(int, struct sockaddr *, socklen_t *) =
      (int (*)(int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "accept");
  if (real_accept == NULL) {
    RM_PRINTF("ERROR: failed to query dlsym for 'accept': %s\n", dlerror());
    return -1;
  }
  int fd = real_accept(sockfd, addr, addrlen);
  if (fd == -1) {
    RM_PRINTF("ERROR: real 'accept' failed\n");
    return fd;
  }
  if (addr != NULL && addr->sa_family != AF_INET) {
    RM_PRINTF("WARNING: got 'accept' for non-AF_INET sa_family=%u\n",
              addr->sa_family);
    if (addr->sa_family == AF_INET6) {
      RM_PRINTF("WARNING: (continued) got 'accept' for AF_INET6\n");
    }
    return fd;
  }

  // If we have been signalled to quit, then do nothing more.
  if (!run)
    return fd;
  // One-time setup.
  lock_setup.lock();
  if (!setup_done) {
    if (!setup()) {
      lock_setup.unlock();
      return fd;
    }
    setup_done = true;
  }
  lock_setup.unlock();
  // Look up the four-tuple.
  struct rm_flow flow;
  if (!get_flow(fd, &flow))
    return fd;
  RM_PRINTF("flow: %u:%u->%u:%u\n", flow.remote_addr, flow.remote_port,
            flow.local_addr, flow.local_port);
  // Ignore flows that are not in the monitor port range.
  if (!(flow.remote_port >= monitor_port_start &&
        flow.remote_port <= monitor_port_end)) {
    RM_PRINTF(
        "INFO: ignoring flow on remote port %u, not in monitor port range: "
        "[%u, %u]\n",
        flow.remote_port, monitor_port_start, monitor_port_end);
    return fd;
  }
  fd_to_flow[fd] = flow;
  // Change the CCA to BPF_CUBIC.
  if (!set_cca(fd, RM_BPF_CUBIC))
    return fd;
  // Initial scheduling for this flow.
  lock_scheduler.lock();
  initial_scheduling(fd);
  lock_scheduler.unlock();
  RM_PRINTF("INFO: successful 'accept' for FD=%d, got FD=%d\n", sockfd, fd);
  return fd;
}

// Get around C++ function name mangling.
extern "C" {
int close(int sockfd) {
  static int (*real_close)(int) = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == NULL) {
    RM_PRINTF("ERROR: failed to query dlsym for 'close': %s\n", dlerror());
    return -1;
  }
  int ret = real_close(sockfd);
  if (ret == -1) {
    RM_PRINTF("ERROR: real 'close' failed\n");
  } else {
    RM_PRINTF("INFO: successful 'close' for FD=%d\n", sockfd);
  }

  // Remove this FD from all data structures.
  lock_scheduler.lock();
  if (fd_to_flow.contains(sockfd)) {
    // Obviously, do this before removing the FD from fd_to_flow.
    if (flow_to_rwnd_fd)
      bpf_map_delete_elem(flow_to_rwnd_fd, &fd_to_flow[sockfd]);
    if (flow_to_win_scale_fd)
      bpf_map_delete_elem(flow_to_win_scale_fd, &fd_to_flow[sockfd]);
    if (flow_to_last_data_time_fd)
      bpf_map_delete_elem(flow_to_last_data_time_fd, &fd_to_flow[sockfd]);
    if (flow_to_keepalive_fd)
      bpf_map_delete_elem(flow_to_keepalive_fd, &fd_to_flow[sockfd]);
    // Removing the FD from fd_to_flow triggers it to be (eventually) removed
    // from scheduling.
    unsigned long d = fd_to_flow.erase(sockfd);
    RM_PRINTF("INFO: removed FD=%d (%ld elements removed)\n", sockfd, d);
  } else {
    RM_PRINTF("INFO: ignoring 'close' for FD=%d, not in fd_to_flow\n", sockfd);
  }
  lock_scheduler.unlock();
  return ret;
}
}

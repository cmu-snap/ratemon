#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <dlfcn.h>
#include <linux/inet_diag.h>
#include <netinet/in.h>  // structure for storing address information
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>  // for socket APIs
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
#include <utility>

#include "ratemon.h"

// Protects writes only to max_active_flows, epoch_us, idle_timeout_us,
// num_to_schedule, monitor_port_start, monitor_port_end, flow_to_rwnd_fd,
// flow_to_win_scale_fd, oldact, and setup. Reads are unprotected.
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
unsigned int idle_timeout_us = 1000;
unsigned int num_to_schedule = 1;
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

// Call this when a flow should be paused. If there are waiting flows and
// available capacity, then one will be activated. Flows are paused and
// activated in round-robin order. Each flow is allowed to be active for
// at most epoch_us microseconds.
void timer_callback(const boost::system::error_code &error) {
  RM_PRINTF("INFO: in timer_callback\n");

  // 0. Perform validity checks.
  // If an error (such as a cancellation) triggered this callback, then abort
  // immediately.
  if (error) {
    RM_PRINTF("ERROR: timer_callback error: %s\n", error.message().c_str());
    return;
  }
  // If the program has been signalled to stop, then exit.
  if (!run) {
    RM_PRINTF("INFO: exiting\n");
    return;
  }
  // If setup has not been performed yet, then we cannot perform scheduling.
  if (!setup_done) {
    if (timer.expires_from_now(one_sec)) {
      RM_PRINTF("ERROR: cancelled timer when should not have!\n");
    }

    timer.async_wait(&timer_callback);
    RM_PRINTF("INFO: not set up (flag)\n");
    return;
  }
  // At this point, max_active_flows, epoch_us, num_to_schedule,
  // flow_to_rwnd_fd, and flow_to_last_data_time_fd should be set. If not,
  // revert to slow check mode.
  if (max_active_flows == 0 || epoch_us == 0 || idle_timeout_us ||
      num_to_schedule == 0 || flow_to_rwnd_fd == 0 ||
      flow_to_last_data_time_fd == 0) {
    RM_PRINTF(
        "ERROR: cannot continue, invalid max_active_flows=%u, epoch_us=%u, "
        "idle_timeout_us=%u, num_to_schedule=%u, flow_to_rwnd_fd=%d, or "
        "flow_to_last_data_time_fd=%d\n",
        max_active_flows, epoch_us, idle_timeout_us, num_to_schedule,
        flow_to_rwnd_fd, flow_to_last_data_time_fd);
    if (timer.expires_from_now(one_sec)) {
      RM_PRINTF("ERROR: cancelled timer when should not have!\n");
    }
    timer.async_wait(&timer_callback);
    RM_PRINTF("INFO: not set up (params)\n");
    return;
  }

  // It is safe to perform scheduling.
  lock_scheduler.lock();
  RM_PRINTF("INFO: performing scheduling\n");

  // 1. If there are no flows, then revert to slow check mode.
  if (active_fds_queue.empty() && paused_fds_queue.empty()) {
    if (timer.expires_from_now(one_sec)) {
      RM_PRINTF("ERROR: cancelled timer when should not have!\n");
    }
    timer.async_wait(&timer_callback);
    lock_scheduler.unlock();
    RM_PRINTF("INFO: no flows\n");
    return;
  }

  // 2. Remove closed flows from both queues.
  // active_fds_queue
  for (unsigned long i = 0; i < active_fds_queue.size(); ++i) {
    auto a = active_fds_queue.front();
    active_fds_queue.pop();
    if (fd_to_flow.contains(a.first)) active_fds_queue.push(a);
  }
  // paused_fds_queue
  for (unsigned long i = 0; i < paused_fds_queue.size(); ++i) {
    auto p = paused_fds_queue.front();
    paused_fds_queue.pop();
    if (fd_to_flow.contains(p)) paused_fds_queue.push(p);
  }

  // 3. Idle timeout. Look through all active flows and pause any that have been
  // idle for longer than idle_timeout_us. Only do this if there are paused
  // flows.
  boost::posix_time::ptime now =
      boost::posix_time::microsec_clock::local_time();
  if (idle_timeout_us > 0 && !active_fds_queue.empty()) {
    for (unsigned long i = 0; i < active_fds_queue.size(); ++i) {
      auto a = active_fds_queue.front();
      active_fds_queue.pop();
      // Look up last active time.
      unsigned long last_data_time_ns;
      if (bpf_map_lookup_elem(flow_to_last_data_time_fd, &fd_to_flow[a.first],
                              &last_data_time_ns)) {
        active_fds_queue.push(a);
        continue;
      }
      boost::posix_time::ptime last_data_time = {
          {1970, 1, 1}, {0, 0, 0, (long)last_data_time_ns}};
      // If the flow has been active within the idle timeout, then keep it.
      if ((now - last_data_time).total_microseconds() < (long)idle_timeout_us) {
        active_fds_queue.push(a);
        continue;
      }
      // Otherwise, pause the flow.
      paused_fds_queue.push(a.first);
      bpf_map_update_elem(flow_to_rwnd_fd, &fd_to_flow[a.first], &zero,
                          BPF_ANY);
      trigger_ack(a.first);
      RM_PRINTF("INFO: paused FD=%d due to idle timeout\n", a.first);
    }
  }

  // 3. Calculate how many flows to activate.
  // Start with the number of free slots.
  unsigned long num_to_activate = max_active_flows - active_fds_queue.size();
  // If the next flow should be scheduled now, then activate at least
  // num_to_schedule flows.
  if (now > active_fds_queue.front().second) {
    num_to_activate = std::max(num_to_activate, (unsigned long)num_to_schedule);
  }
  // Finally, do not activate more flows than are paused.
  num_to_activate = std::min(num_to_activate, paused_fds_queue.size());

  // 4. Activate the prescribed number of flows.
  boost::posix_time::ptime now_plus_epoch =
      now + boost::posix_time::microseconds(epoch_us);
  for (unsigned long i = 0; i < num_to_activate; ++i) {
    int to_activate = paused_fds_queue.front();
    paused_fds_queue.pop();
    // Randomly jitter the epoch time by +/- 12.5%.
    int rand =
        std::experimental::randint(0, (int)std::roundl(epoch_us * 0.25)) -
        (int)std::roundl(epoch_us * 0.125);
    active_fds_queue.push(
        {to_activate, now_plus_epoch + boost::posix_time::microseconds(rand)});
    bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[to_activate]));
    trigger_ack(to_activate);
    RM_PRINTF("INFO: activated FD=%d\n", to_activate);
  }

  // 4. Pause excessive flows.
  while (active_fds_queue.size() > max_active_flows) {
    int to_pause = active_fds_queue.front().first;
    active_fds_queue.pop();
    if (active_fds_queue.size() < max_active_flows &&
        paused_fds_queue.empty()) {
      // If there are fewer than the limit flows active and there are no
      // waiting flows, then schedule this flow again. Randomly jitter the epoch
      // time by +/- 12.5%.
      int rand =
          std::experimental::randint(0, (int)std::roundl(epoch_us * 0.25)) -
          (int)std::roundl(epoch_us * 0.125);
      boost::posix_time::ptime now_plus_epoch_plus_rand =
          now_plus_epoch + boost::posix_time::microseconds(rand);
      active_fds_queue.push({to_pause, now_plus_epoch_plus_rand});
      RM_PRINTF("INFO: reactivated FD=%d\n", to_pause);
    } else {
      // Pause this flow.
      paused_fds_queue.push(to_pause);
      bpf_map_update_elem(flow_to_rwnd_fd, &(fd_to_flow[to_pause]), &zero,
                          BPF_ANY);
      trigger_ack(to_pause);
      RM_PRINTF("INFO: paused FD=%d\n", to_pause);
    }
  }

  // 5. Check invariants.
  // Cannot have more than the max number of active flows.
  assert(active_fds_queue.size() <= max_active_flows);
  // If there are no active flows, then there should also be no paused flows.
  assert(!active_fds_queue.empty() || paused_fds_queue.empty());

  // 6. Calculate when the next timer should expire.
  boost::posix_time::time_duration when;
  if (active_fds_queue.empty()) {
    // If there are no flows, revert to slow check mode.
    RM_PRINTF("INFO: no flows remaining\n");
    when = one_sec;
  } else if (idle_timeout_us == 0) {
    // If we are using idle timeout mode...
    when = boost::posix_time::microsec(
        std::min((long)idle_timeout_us,
                 (active_fds_queue.front().second - now).total_microseconds()));
  } else {
    // If we are not using idle timeout mode...
    when = active_fds_queue.front().second - now;
  }

  // 7. Start the next timer.
  if (timer.expires_from_now(one_sec)) {
    RM_PRINTF("ERROR: cancelled timer when should not have!\n");
  }
  timer.async_wait(&timer_callback);
  lock_scheduler.unlock();
  RM_PRINTF("INFO: sleeping until next event\n");
  return;
}

void thread_func() {
  // This function is designed to be run in a thread. It is responsible for
  // managing the async timers that perform scheduling.
  RM_PRINTF("INFO: scheduler thread started\n");
  if (timer.expires_from_now(one_sec)) {
    RM_PRINTF("ERROR: cancelled timer when should not have!\n");
  }

  timer.async_wait(&timer_callback);
  RM_PRINTF("INFO: scheduler thread initial sleep\n");
  // Execute the configured events, until there are no more events to execute.
  io.run();

  // Delete all flows from flow_to_rwnd and flow_to_win_scale.
  lock_scheduler.lock();
  for (const auto &p : fd_to_flow) {
    if (flow_to_rwnd_fd != 0) bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);

    if (flow_to_win_scale_fd != 0)
      bpf_map_delete_elem(flow_to_win_scale_fd, &p.second);

    if (flow_to_last_data_time_fd != 0)
      bpf_map_delete_elem(flow_to_last_data_time_fd, &p.second);
  }
  lock_scheduler.unlock();
  RM_PRINTF("INFO: scheduler thread ended\n");

  if (run) {
    RM_PRINTF(
        "ERROR: scheduled thread ended before program was signalled to "
        "stop!\n");
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
  if (!read_env_uint(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows)) return false;
  if (!read_env_uint(RM_EPOCH_US_KEY, &epoch_us, true /* allow_zero */))
    return false;
  if (!read_env_uint(RM_IDLE_TIMEOUT_US_KEY, &idle_timeout_us)) return false;
  // Cannot schedule more than max_active_flows at once.
  if (!read_env_uint(RM_NUM_TO_SCHEDULE_KEY, &num_to_schedule) ||
      num_to_schedule > max_active_flows)
    return false;
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
  RM_PRINTF("INFO: successfully looked up 'flow_to_rwnd' FD\n");

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
  RM_PRINTF("INFO: successfully looked up 'flow_to_win_scale' FD\n");

  // Look up the FD for the flow_to_last_data_time_ns map. We do not need the
  // BPF skeleton for this.
  err = bpf_obj_get(RM_FLOW_TO_LAST_DATA_TIME_PIN_PATH);
  if (err == -1) {
    RM_PRINTF(
        "ERROR: failed to get FD for 'flow_to_last_data_time_ns' from path "
        "'%s'\n",
        RM_FLOW_TO_RWND_PIN_PATH);
    return false;
  }
  flow_to_last_data_time_fd = err;
  RM_PRINTF("INFO: successfully looked up 'flow_to_last_data_time_ns' FD\n");

  // Catch SIGINT to end the program.
  struct sigaction action;
  action.sa_handler = sigint_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, &oldact);

  // Launch the scheduler thread.
  scheduler_thread = boost::thread(thread_func);

  RM_PRINTF(
      "INFO: setup complete! max_active_flows=%u, epoch_us=%u, "
      "idle_timeout_us=%u, "
      "num_to_schedule=%u, monitor_port_start=%u, monitor_port_end=%u\n",
      max_active_flows, epoch_us, idle_timeout_us, num_to_schedule,
      monitor_port_start, monitor_port_end);
  return true;
}

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

bool set_cca(int fd, const char *cca) {
  // Set the CCA and make sure it was set correctly.
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
void initial_scheduling(int fd, struct rm_flow *flow) {
  // Should this flow be active or paused?
  if (active_fds_queue.size() < max_active_flows) {
    // Less than the max number of flows are active, so make this one active.
    boost::posix_time::ptime now =
        boost::posix_time::microsec_clock::local_time();
    int rand =
        std::experimental::randint(0, (int)std::roundl(epoch_us * 0.25)) -
        (int)std::roundl(epoch_us * 0.125);
    active_fds_queue.push({fd, now + boost::posix_time::microseconds(epoch_us) +
                                   boost::posix_time::microseconds(rand)});
    RM_PRINTF("INFO: allowing new flow FD=%d\n", fd);
    if (active_fds_queue.size() == 1) {
      if (timer.expires_from_now(active_fds_queue.front().second - now) != 1) {
        RM_PRINTF("ERROR: should have cancelled 1 timer!\n");
      }

      timer.async_wait(&timer_callback);
      RM_PRINTF("INFO: first scheduling event\n");
    }
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds_queue.push(fd);
    // Pausing a flow means retting its RWND to 0 B.
    bpf_map_update_elem(flow_to_rwnd_fd, flow, &zero, BPF_ANY);
    RM_PRINTF("INFO: paused new flow FD=%d\n", fd);
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
      RM_PRINTF("WARNING: (continued) got 'accept' for AF_INET6!\n");
    }

    return fd;
  }

  // If we have been signalled to quit, then do nothing more.
  if (!run) return fd;
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
  if (!get_flow(fd, &flow)) return fd;
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
  if (!set_cca(fd, RM_BPF_CUBIC)) return fd;
  // Initial scheduling for this flow.
  lock_scheduler.lock();
  initial_scheduling(fd, &flow);
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

  // Remove this FD from all data structures.
  lock_scheduler.lock();
  if (fd_to_flow.contains(sockfd)) {
    // Obviously, do this before removing the FD from fd_to_flow.
    if (flow_to_rwnd_fd != 0)
      bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[sockfd]));

    if (flow_to_win_scale_fd != 0)
      bpf_map_delete_elem(flow_to_win_scale_fd, &(fd_to_flow[sockfd]));

    if (flow_to_last_data_time_fd != 0)
      bpf_map_delete_elem(flow_to_last_data_time_fd, &(fd_to_flow[sockfd]));

    // Removing the FD from fd_to_flow triggers it to be (eventually) removed
    // from scheduling.
    fd_to_flow.erase(sockfd);
    RM_PRINTF("INFO: removed FD=%d\n", sockfd);
  } else {
    RM_PRINTF("INFO: ignoring 'close' for FD=%d, not in fd_to_flow\n", sockfd);
  }

  lock_scheduler.unlock();

  if (ret == -1) {
    RM_PRINTF("ERROR: real 'close' failed\n");
    return ret;
  }
  RM_PRINTF("INFO: successful 'close' for FD=%d\n", sockfd);
  return ret;
}
}

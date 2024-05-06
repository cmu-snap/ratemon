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
#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>

#include "ratemon.h"

// Protects writes only to max_active_flows, epoch_us, num_to_schedule,
// monitor_port, flow_to_rwnd_fd, flow_to_win_scale_fd, oldact, and setup. Reads
// are unprotected.
std::mutex lock_setup;
// Whether setup has been performed.
bool setup = false;
// Used to signal the scheduler thread to end.
bool run = true;
// Existing signal handler for SIGINT.
struct sigaction oldact;
// FD for the BPF map "flow_to_rwnd".
int flow_to_rwnd_fd = 0;
// FD for the BPF map "flow_to_win_sca" (short for "flow_to_win_scale").
int flow_to_win_scale_fd = 0;
// Manages the io_service.
boost::thread scheduler_thread;
// Runs async timers for scheduling
boost::asio::io_service io;
// Periodically performs scheduling using timer_callback().
boost::asio::deadline_timer timer(io);
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
unsigned int num_to_schedule = 1;
unsigned short monitor_port = 9000;

// Used to set entries in flow_to_rwnd.
int zero = 0;
boost::posix_time::seconds one_sec = boost::posix_time::seconds(1);
// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length = (socklen_t)sizeof(placeholder_cc_info);

inline void trigger_ack(int fd) {
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&placeholder_cc_info,
             &placeholder_cc_info_length);
}

void timer_callback(const boost::system::error_code &error) {
  // Call this when a flow should be paused. Only one flow will be paused at a
  // time. If there are waiting flows, one will be activated. Flows are paused
  // and activated in round-robin order. Each flow is allowed to be active for
  // epoch_us microseconds.
  if (error) {
    RM_PRINTF("ERROR: timer_callback error: %s\n", error.message().c_str());
    return;
  }

  if (!run) {
    RM_PRINTF("INFO: exiting...\n");
    return;
  }

  // If setup has not been performed yet, then we cannot perform scheduling.
  if (!setup) {
    // RM_PRINTF("WARNING setup not completed, skipping scheduling\n");
    timer.expires_from_now(one_sec);
    timer.async_wait(&timer_callback);
    return;
  }
  // At this point, max_active_flows, epoch_us, num_to_schedule, and
  // flow_to_rwnd_fd should be set.
  if (max_active_flows == 0 || epoch_us == 0 || num_to_schedule == 0 ||
      flow_to_rwnd_fd == 0) {
    RM_PRINTF(
        "ERROR: cannot continue, invalid max_active_flows=%u, epoch_us=%u, "
        "num_to_schedule=%u, or "
        "flow_to_rwnd_fd=%d\n",
        max_active_flows, epoch_us, num_to_schedule, flow_to_rwnd_fd);
    timer.expires_from_now(one_sec);
    timer.async_wait(&timer_callback);
    return;
  }

  lock_scheduler.lock();
  RM_PRINTF("Performing scheduling\n");

  if (active_fds_queue.empty()) {
    timer.expires_from_now(one_sec);
    timer.async_wait(&timer_callback);
    lock_scheduler.unlock();
    return;
  }

  boost::posix_time::ptime now =
      boost::posix_time::microsec_clock::local_time();
  boost::posix_time::ptime now_plus_epoch =
      now + boost::posix_time::microseconds(epoch_us);

  if (now < active_fds_queue.front().second) {
    // The next flow should not be scheduled yet.
    timer.expires_at(active_fds_queue.front().second);
    timer.async_wait(&timer_callback);
    lock_scheduler.unlock();
    return;
  }

  // Need to keep track of the number of active flows before we do any
  // scheduling so that we do not accidentally pause flows that we just
  // activated.
  auto num_active = active_fds_queue.size();

  // Schedule up to num_to_schedule flows, one at a time. But if there are not
  // enough flows, then schedule the max number in either queue. One iteration
  // of this loop can service one flow from each queue.
  for (unsigned int i = 0;
       i < std::min((long unsigned int)num_to_schedule,
                    std::max(active_fds_queue.size(), paused_fds_queue.size()));
       ++i) {
    // If there are paused flows, then activate one. Loop until we find a paused
    // flow that is valid, meaning that it has not been removed from fd_to_flow
    // (i.e., has not been close()'ed).
    while (!paused_fds_queue.empty()) {
      int to_activate = paused_fds_queue.front();
      paused_fds_queue.pop();
      if (fd_to_flow.contains(to_activate)) {
        // This flow is valid, so activate it. Record the time at which its
        // epoch is over.
        active_fds_queue.push({to_activate, now_plus_epoch});
        bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[to_activate]));
        trigger_ack(to_activate);
        RM_PRINTF("INFO: activated FD=%d\n", to_activate);
        break;
      }
    }

    // Do not pause more active flows than were active at the start of this
    // epoch, otherwise we will be pausing flows that we just activated.
    if (i < num_active) {
      // We always need to pop the front of the active_fds_queue because it was
      // that flow's timer which triggered the current scheduling event. We
      // either schedule that flow again or pause it.
      int to_pause = active_fds_queue.front().first;
      active_fds_queue.pop();
      if (fd_to_flow.contains(to_pause)) {
        if (active_fds_queue.size() < max_active_flows &&
            paused_fds_queue.empty()) {
          // If there are fewer than the limit flows active and there are no
          // waiting flows, then scheduling this flow again. But first, check to
          // make sure that it is still valid (see above).
          active_fds_queue.push({to_pause, now_plus_epoch});
          RM_PRINTF("INFO: reactivated FD=%d\n", to_pause);
        } else {
          // Pause this flow.
          paused_fds_queue.push(to_pause);
          bpf_map_update_elem(flow_to_rwnd_fd, &(fd_to_flow[to_pause]), &zero,
                              BPF_ANY);
          trigger_ack(to_pause);
          RM_PRINTF("INFO: paused FD=%d\n", to_pause);
        }
      } else {
        RM_PRINTF("INFO: cleaned up FD=%d\n", to_pause);
      }
    }
  }

  // Invariants.
  // Cannot have more than the max number of active flows.
  assert(active_fds_queue.size() <= max_active_flows);
  // If there are no active flows, then there should also be no paused flows.
  assert(!active_fds_queue.empty() || paused_fds_queue.empty());

  // Schedule the next timer callback for when the next flow should be paused.
  if (active_fds_queue.empty()) {
    // If we cleaned up all flows, then revert to slow check mode.
    timer.expires_from_now(one_sec);
    timer.async_wait(&timer_callback);
    lock_scheduler.unlock();
    return;
  }
  // Trigger scheduling for the next flow.
  timer.expires_at(active_fds_queue.front().second);
  timer.async_wait(&timer_callback);
  lock_scheduler.unlock();
  return;
}

void thread_func() {
  // This function is designed to be run in a thread. It is responsible for
  // managing the async timers that perform scheduling.
  RM_PRINTF("Scheduler thread started\n");
  timer.expires_from_now(one_sec);
  timer.async_wait(&timer_callback);
  // Execute the configured events, until there are no more events to execute.
  io.run();
  RM_PRINTF("Scheduler thread ended\n");
}

// Catch SIGINT and trigger the main function to end.
void sigint_handler(int signum) {
  switch (signum) {
    case SIGINT:
      RM_PRINTF("INFO: caught SIGINT\n");
      run = false;
      scheduler_thread.join();
      RM_PRINTF("Resetting old SIGINT handler\n");
      sigaction(SIGINT, &oldact, NULL);
      break;
    default:
      RM_PRINTF("ERROR: caught signal %d\n", signum);
      break;
  }
  RM_PRINTF("INFO: re-raising signal %d\n", signum);
  raise(signum);
}

bool read_env_uint(const char *key, volatile unsigned int *dest) {
  // Read an environment variable as an unsigned integer.
  char *val_str = getenv(key);
  if (val_str == NULL) {
    RM_PRINTF("ERROR: failed to query environment variable '%s'\n", key);
    return false;
  }
  int val_int = atoi(val_str);
  if (val_int <= 0) {
    RM_PRINTF("ERROR: invalid value for '%s'=%d (must be > 0)\n", key, val_int);
    return false;
  }
  *dest = (unsigned int)val_int;
  return true;
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
  int new_fd = real_accept(sockfd, addr, addrlen);
  if (new_fd == -1) {
    RM_PRINTF("ERROR: real 'accept' failed\n");
    return new_fd;
  }
  if (addr != NULL && addr->sa_family != AF_INET) {
    RM_PRINTF("WARNING: got 'accept' for non-AF_INET sa_family=%u\n",
              addr->sa_family);
    if (addr->sa_family == AF_INET6) {
      RM_PRINTF("WARNING: (continued) got 'accept' for AF_INET6!\n");
    }
    return new_fd;
  }

  // Perform setup (only once for all flows in this process), such as reading
  // parameters from environment variables and looking up the BPF map
  // flow_to_rwnd.
  lock_setup.lock();
  if (!setup) {
    if (!read_env_uint(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows)) {
      lock_setup.unlock();
      return new_fd;
    }
    if (!read_env_uint(RM_EPOCH_US_KEY, &epoch_us)) {
      lock_setup.unlock();
      return new_fd;
    }
    // Cannot schedule more than max_active_flows at once.
    if (!read_env_uint(RM_NUM_TO_SCHEDULE_KEY, &num_to_schedule) ||
        num_to_schedule > max_active_flows) {
      lock_setup.unlock();
      return new_fd;
    }
    unsigned int monitor_port_;
    if (!read_env_uint(RM_MONITOR_PORT, &monitor_port_) ||
        monitor_port_ >= 65536) {
      lock_setup.unlock();
      return new_fd;
    }
    monitor_port = (unsigned short)monitor_port_;

    // Look up the FD for the flow_to_rwnd map. We do not need the BPF skeleton
    // for this.
    int err = bpf_obj_get(RM_FLOW_TO_RWND_PIN_PATH);
    if (err == -1) {
      RM_PRINTF("ERROR: failed to get FD for 'flow_to_rwnd' from path '%s'\n",
                RM_FLOW_TO_RWND_PIN_PATH);
      lock_setup.unlock();
      return new_fd;
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
      lock_setup.unlock();
      return new_fd;
    }
    flow_to_win_scale_fd = err;
    RM_PRINTF("INFO: successfully looked up 'flow_to_win_scale' FD\n");

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
        "num_to_schedule=%u, monitor_port=%u\n",
        max_active_flows, epoch_us, num_to_schedule, monitor_port);
    setup = true;
  }
  lock_setup.unlock();

  // Determine the four-tuple, which we need to track because RWND tuning is
  // applied based on four-tuple.
  struct sockaddr_in local_addr;
  socklen_t local_addr_len = sizeof(local_addr);
  // Get the local IP and port.
  if (getsockname(new_fd, (struct sockaddr *)&local_addr, &local_addr_len) ==
      -1) {
    RM_PRINTF("ERROR: failed to call 'getsockname'\n");
    return -1;
  }
  struct sockaddr_in remote_addr;
  socklen_t remote_addr_len = sizeof(remote_addr);
  // Get the peer's (i.e., the remote) IP and port.
  if (getpeername(new_fd, (struct sockaddr *)&remote_addr, &remote_addr_len) ==
      -1) {
    RM_PRINTF("ERROR: failed to call 'getpeername'\n");
    return -1;
  }
  // Fill in the four-tuple.
  struct rm_flow flow = {.local_addr = ntohl(local_addr.sin_addr.s_addr),
                         .remote_addr = ntohl(remote_addr.sin_addr.s_addr),
                         .local_port = ntohs(local_addr.sin_port),
                         .remote_port = ntohs(remote_addr.sin_port)};
  RM_PRINTF("flow: %u:%u->%u:%u\n", flow.remote_addr, flow.remote_port,
            flow.local_addr, flow.local_port);

  if (flow.local_port != monitor_port) {
    RM_PRINTF("INFO: ignoring flow on port %u, not on monitor port %u\n",
              flow.local_port, monitor_port);
    return new_fd;
  }

  // Set the CCA and make sure it was set correctly.
  if (setsockopt(new_fd, SOL_TCP, TCP_CONGESTION, RM_BPF_CUBIC,
                 strlen(RM_BPF_CUBIC)) == -1) {
    RM_PRINTF("ERROR: failed to 'setsockopt' TCP_CONGESTION\n");
    return new_fd;
  }
  char retrieved_cca[32];
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(new_fd, SOL_TCP, TCP_CONGESTION, retrieved_cca,
                 &retrieved_cca_len) == -1) {
    RM_PRINTF("ERROR: failed to 'getsockopt' TCP_CONGESTION\n");
    return new_fd;
  }
  if (strcmp(retrieved_cca, RM_BPF_CUBIC)) {
    RM_PRINTF("ERROR: failed to set CCA to %s! Actual CCA is: %s\n",
              RM_BPF_CUBIC, retrieved_cca);
    return new_fd;
  }

  lock_scheduler.lock();
  fd_to_flow[new_fd] = flow;
  // Should this flow be active or paused?
  if (active_fds_queue.size() < max_active_flows) {
    // Less than the max number of flows are active, so make this one active.
    active_fds_queue.push(
        {new_fd, boost::posix_time::microsec_clock::local_time() +
                     boost::posix_time::microseconds(epoch_us)});
    RM_PRINTF("INFO: allowing new flow FD=%d\n", new_fd);
    if (active_fds_queue.size() == 1) {
      timer.cancel();
      timer.expires_at(active_fds_queue.front().second);
      timer.async_wait(&timer_callback);
    }
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds_queue.push(new_fd);
    // Pausing a flow means retting its RWND to 0 B.
    bpf_map_update_elem(flow_to_rwnd_fd, &flow, &zero, BPF_ANY);
    RM_PRINTF("INFO: paused new flow FD=%d\n", new_fd);
  }
  lock_scheduler.unlock();

  RM_PRINTF("INFO: successful 'accept' for FD=%d, got FD=%d\n", sockfd, new_fd);
  return new_fd;
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
    if (flow_to_rwnd_fd != 0) {
      bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[sockfd]));
    }
    if (flow_to_win_scale_fd != 0) {
      bpf_map_delete_elem(flow_to_win_scale_fd, &(fd_to_flow[sockfd]));
    }
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
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>  // for socket APIs
#include <unistd.h>

#include <boost/thread.hpp>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ratemon.h"

std::mutex lock_control;
unsigned int max_active_flows = 0;
unsigned int epoch_us = 0;
bool setup = false;
bool skipped_first = false;
// FD for the flow_to_rwnd map.
int flow_to_rwnd_fd = 0;

// Protects active_fds_queue and paused_fds_queue.
std::mutex lock_scheduler;
std::queue<int> active_fds_queue;
std::queue<int> paused_fds_queue;
// Maps file descriptor to rm_flow struct.
std::unordered_map<int, struct rm_flow> fd_to_flow;

// Used to set entries in flow_to_rwnd.
int zero = 0;

// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length = (socklen_t)sizeof(placeholder_cc_info);

inline void trigger_ack(int fd) {
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&placeholder_cc_info,
             &placeholder_cc_info_length);
}

void thread_func() {
  RM_PRINTF("INFO: libratemon_interp scheduling thread started\n");
  // If setup has not been performed yet, then we cannot perform scheduling.
  while (!setup) {
    // RM_PRINTF("WARNING setup not completed, skipping scheduling\n");
    usleep(10000);
  }
  // At this point, max_active_flows, epoch_us, and flow_to_rwnd_fd should be.
  if (max_active_flows == 0 || epoch_us == 0 || flow_to_rwnd_fd == 0) {
    lock_control.unlock();
    RM_PRINTF(
        "ERROR: cannot continue, invalid max_active_flows=%u, epoch_us=%u, or "
        "flow_to_rwnd_fd=%d\n",
        max_active_flows, epoch_us, flow_to_rwnd_fd);
    return;
  }

  // Previously paused flows that will be activated.
  std::vector<int> new_active_fds;
  // Preallocate suffient space.
  new_active_fds.reserve(max_active_flows);

  while (true) {
    usleep(epoch_us);

    // If fewer than the max number of flows exist and they are all active, then
    // there is no need for scheduling.
    if (active_fds_queue.size() < max_active_flows &&
        paused_fds_queue.size() == 0) {
      // RM_PRINTF("WARNING insufficient flows, skipping scheduling\n");
      continue;
    }

    RM_PRINTF("Performing scheduling\n");
    new_active_fds.clear();
    lock_scheduler.lock();

    // Try to find max_active_flows FDs to unpause.
    while (!paused_fds_queue.empty() and
           new_active_fds.size() < max_active_flows) {
      // If we still know about this flow, then we can activate it.
      int next_fd = paused_fds_queue.front();
      if (fd_to_flow.contains(next_fd)) {
        paused_fds_queue.pop();
        new_active_fds.push_back(next_fd);
      }
    }
    auto num_prev_active = active_fds_queue.size();

    // For each of the flows chosen to be activated, add it to the active set
    // and remove it from the RWND map. Trigger an ACK to wake it up. Note that
    // twice the allowable number of flows will be active briefly.
    RM_PRINTF("Activating %lu flows: ", new_active_fds.size());
    for (const auto &fd : new_active_fds) {
      RM_PRINTF("%d ", fd);
      active_fds_queue.push(fd);
      bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[fd]));
      trigger_ack(fd);
    }
    RM_PRINTF("\n");

    // For each fo the previously active flows, add it to the paused set,
    // install an RWND mapping to actually pause it, and trigger an ACK to
    // communicate the new RWND value.
    RM_PRINTF("Pausing %lu flows: ", num_prev_active);
    for (unsigned long i = 0; i < num_prev_active; i++) {
      int fd = active_fds_queue.front();
      active_fds_queue.pop();
      RM_PRINTF("%d ", fd);
      paused_fds_queue.push(fd);
      bpf_map_update_elem(flow_to_rwnd_fd, &(fd_to_flow[fd]), &zero, BPF_ANY);
      // TODO: Do we need to send an ACK to immediately pause the flow?
      trigger_ack(fd);
    }
    RM_PRINTF("\n");

    lock_scheduler.unlock();
    new_active_fds.clear();
  }
}

boost::thread t(thread_func);

bool read_env_uint(const char *key, volatile unsigned int *dest) {
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

  // Perform BPF setup (only once for all flows in this process).
  lock_control.lock();
  if (!setup) {
    if (!read_env_uint(RM_MAX_ACTIVE_FLOWS_KEY, &max_active_flows)) {
      return new_fd;
    }
    if (!read_env_uint(RM_EPOCH_US_KEY, &epoch_us)) {
      return new_fd;
    }

    int err = bpf_obj_get(RM_FLOW_TO_RWND_PIN_PATH);
    if (err == -1) {
      RM_PRINTF("ERROR: failed to get FD for 'flow_to_rwnd'\n");
      return new_fd;
    }
    flow_to_rwnd_fd = err;
    RM_PRINTF("INFO: successfully looked up 'flow_to_rwnd' FD\n");
    setup = true;

    RM_PRINTF("INFO: max_active_flows=%u epoch_us=%u\n", max_active_flows,
              epoch_us);
  }
  lock_control.unlock();

  // Hack for iperf3. The first flow is a control flow that should not be
  // scheduled. Note that for this hack to work, libratemon_interp must be
  // restarted between tests.
  lock_control.lock();
  if (fd_to_flow.size() == 0 && !skipped_first) {
    RM_PRINTF("WARNING: skipping first flow\n");
    skipped_first = true;
    lock_control.unlock();
    return new_fd;
  }
  lock_control.unlock();

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
  // RM_PRINTF("flow: %u:%u->%u:%u\n", flow.remote_addr, flow.remote_port,
  //           flow.local_addr, flow.local_port);

  lock_scheduler.lock();
  fd_to_flow[new_fd] = flow;
  // Should this flow be active or paused?
  if (active_fds_queue.size() < max_active_flows) {
    // Less than the max number of flows are active, so make this one active.
    active_fds_queue.push(new_fd);
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds_queue.push(new_fd);
    // Pausing a flow means retting its RWND to 0 B.
    bpf_map_update_elem(flow_to_rwnd_fd, &flow, &zero, BPF_ANY);
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
  if (ret == -1) {
    RM_PRINTF("ERROR: real 'close' failed\n");
    return ret;
  }

  // To get the flow struct for this FD, we must use visit() to look it up
  // in the concurrent_flat_map. Obviously, do this before removing the FD
  // from fd_to_flow.
  bpf_map_delete_elem(flow_to_rwnd_fd, &(fd_to_flow[sockfd]));
  // Removing the FD from fd_to_flow triggers it to be (eventually) removed from
  // scheduling.
  fd_to_flow.erase(sockfd);

  RM_PRINTF("INFO: successful 'close' for FD=%d\n", sockfd);
  return ret;
}
}
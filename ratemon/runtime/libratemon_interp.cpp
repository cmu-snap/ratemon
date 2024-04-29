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
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/unordered/concurrent_flat_set.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ratemon.h"
#include "ratemon.skel.h"

//
bool setup = false;

// BPF things.
struct ratemon_bpf *skel = NULL;
struct bpf_map *flow_to_rwnd = NULL;
int flow_to_rwnd_fd = 0;

// File descriptors of flows that are allowed to send and therefore do not have
// an entry in flow_to_rwnd.
boost::unordered::concurrent_flat_set<int> active_fds;
// File descriptors of flows that are paused and therefore have an entry of 0 B
// in flow_to_rwnd.
boost::unordered::concurrent_flat_set<int> paused_fds;
// Maps file descriptor to flow struct.
boost::unordered::concurrent_flat_map<int, struct rm_flow> fd_to_flow;

// Used to set entries in flow_to_rwnd.
int zero = 0;

// As an optimization, reuse the same tcp_cc_info struct and size.
union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length = (socklen_t)sizeof(placeholder_cc_info);

// Look up the environment variable for max active flows.
unsigned int max_active_flows =
    (unsigned int)atoi(getenv(RM_MAX_ACTIVE_FLOWS_KEY));

// Look up the environment variable for scheduling epoch.
unsigned int epoch_us = (unsigned int)atoi(getenv(RM_EPOCH_US_KEY));

inline void trigger_ack(int fd) {
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&placeholder_cc_info,
             &placeholder_cc_info_length);
}

void thread_func() {
  // Variables used during each scheduling epoch.
  // Currently active flows, soon to be previously active.
  std::vector<int> prev_active_fds;
  // Length of prev_active_fds, which is the number of flows to be unpaused.
  size_t num_to_find = 0;
  // Previously paused flows that will be activated.
  std::vector<int> new_active_fds;
  // Preallocate suffient space.
  prev_active_fds.reserve(max_active_flows);
  new_active_fds.reserve(max_active_flows);

  if (max_active_flows == 0 || epoch_us == 0) {
    RM_PRINTF("ERROR when querying environment variables '%s' or '%s'\n",
              RM_MAX_ACTIVE_FLOWS_KEY, RM_EPOCH_US_KEY);
    return;
  }

  RM_PRINTF(
      "libratemon_interp scheduling thread started, max flows=%u, epoch=%u "
      "us\n",
      max_active_flows, epoch_us);

  while (true) {
    usleep(epoch_us);
    // RM_PRINTF("Time to schedule\n");

    // If setup has not been performed yet, then we cannot perform scheduling.
    if (!setup) {
      // RM_PRINTF("WARNING setup not completed, skipping scheduling\n");
      continue;
    }

    // If fewer than the max number of flows exist and they are all active, then
    // there is no need for scheduling.
    if (active_fds.size() < max_active_flows && paused_fds.size() == 0) {
      // RM_PRINTF("WARNING insufficient flows, skipping scheduling\n");
      continue;
    }

    // RM_PRINTF("Performing scheduling\n");

    // Make a copy of the currently (soon-to-be previously) active flows.
    active_fds.visit_all([&](const int &fd) { prev_active_fds.push_back(fd); });

    // For each previously active flow, try to find a paused flow to replace it.
    num_to_find = prev_active_fds.size();
    paused_fds.visit_while([&](const int &x) {
      if (new_active_fds.size() < num_to_find) {
        new_active_fds.push_back(x);
        return true;
      } else {
        return false;
      }
    });

    // For each of the flows chosen to be activated, add it to the active set
    // and remove it from both the paused set and the RWND map. Trigger an ACK
    // to wake it up. Note that twice the allowable number of flows will be
    // active briefly.
    RM_PRINTF("Activating %lu flows: ", new_active_fds.size());
    for (const auto &fd : new_active_fds) {
      RM_PRINTF("%d ", fd);
      active_fds.insert(fd);
      paused_fds.erase(fd);
      fd_to_flow.visit(fd, [](const auto &p) {
        bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);
      });
      trigger_ack(fd);
    }
    RM_PRINTF("\n");

    // For each fo the previously active flows, add it to the paused set, remove
    // it from the active set, and install an RWND mapping to actually pause it.
    RM_PRINTF("Pausing %lu flows: ", prev_active_fds.size());
    for (const auto &fd : prev_active_fds) {
      RM_PRINTF("%d ", fd);
      paused_fds.insert(fd);
      active_fds.erase(fd);
      fd_to_flow.visit(fd, [](const auto &p) {
        bpf_map_update_elem(flow_to_rwnd_fd, &p.second, &zero, BPF_ANY);
      });
    }
    RM_PRINTF("\n");

    // Clear temporary data structures.
    prev_active_fds.clear();
    new_active_fds.clear();

    fflush(stdout);
  }
}

boost::thread t(thread_func);

// For some reason, C++ function name mangling does not prevent us from
// overriding accept(), so we do not need 'extern "C"'.
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  if (max_active_flows == 0 || epoch_us == 0) {
    RM_PRINTF("ERROR when querying environment variables '%s' or '%s'\n",
              RM_MAX_ACTIVE_FLOWS_KEY, RM_EPOCH_US_KEY);
    return -1;
  }

  static int (*real_accept)(int, struct sockaddr *, socklen_t *) =
      (int (*)(int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "accept");
  if (real_accept == NULL) {
    RM_PRINTF("ERROR when querying dlsym for 'accept': %s\n", dlerror());
    return -1;
  }
  int new_fd = real_accept(sockfd, addr, addrlen);
  if (new_fd == -1) {
    RM_PRINTF("ERROR in real 'accept'\n");
    return new_fd;
  }
  if (addr != NULL && addr->sa_family != AF_INET) {
    RM_PRINTF("WARNING got 'accept' for non-AF_INET: sa_family=%u\n",
              addr->sa_family);
    if (addr->sa_family == AF_INET6) {
      RM_PRINTF("WARNING (continued) got 'accept' for AF_INET6!\n");
    }
    return new_fd;
  }

  // Perform BPF setup (only once for all flows in this process).
  if (!setup) {
    skel = ratemon_bpf__open();
    if (skel == NULL) {
      RM_PRINTF("ERROR when opening BPF skeleton\n");
      return new_fd;
    }

    int pinned_map_fd = bpf_obj_get(RM_FLOW_TO_RWND_PIN_PATH);
    int err = bpf_map__reuse_fd(skel->maps.flow_to_rwnd, pinned_map_fd);
    if (err) {
      RM_PRINTF("ERROR when reusing map FD\n");
      return new_fd;
    }

    flow_to_rwnd = skel->maps.flow_to_rwnd;
    flow_to_rwnd_fd = pinned_map_fd;
    RM_PRINTF("Successfully reused map FD\n");
    setup = true;
  }

  // Set the CCA and make sure it was set correctly.
  if (setsockopt(new_fd, SOL_TCP, TCP_CONGESTION, RM_BPF_CUBIC,
                 strlen(RM_BPF_CUBIC)) == -1) {
    RM_PRINTF("ERROR in 'setsockopt' TCP_CONGESTION\n");
    return new_fd;
  }
  char retrieved_cca[32];
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(new_fd, SOL_TCP, TCP_CONGESTION, retrieved_cca,
                 &retrieved_cca_len) == -1) {
    RM_PRINTF("ERROR in 'getsockopt' TCP_CONGESTION\n");
    return new_fd;
  }
  if (strcmp(retrieved_cca, RM_BPF_CUBIC)) {
    RM_PRINTF("ERROR when setting CCA to %s! Actual CCA is: %s\n", RM_BPF_CUBIC,
              retrieved_cca);
    return new_fd;
  }

  // Determine the four-tuple, which we need to track because RWND tuning is
  // applied based on four-tuple.
  struct sockaddr_in local_addr;
  socklen_t local_addr_len = sizeof(local_addr);
  // Get the local IP and port.
  if (getsockname(new_fd, (struct sockaddr *)&local_addr, &local_addr_len) ==
      -1) {
    RM_PRINTF("ERROR when calling 'getsockname'\n");
    return -1;
  }
  struct sockaddr_in remote_addr;
  socklen_t remote_addr_len = sizeof(remote_addr);
  // Get the peer's (i.e., the remote) IP and port.
  if (getpeername(new_fd, (struct sockaddr *)&remote_addr, &remote_addr_len) ==
      -1) {
    RM_PRINTF("ERROR when calling 'getpeername'\n");
    return -1;
  }
  // Fill in the four-tuple.
  struct rm_flow flow = {.local_addr = ntohl(local_addr.sin_addr.s_addr),
                         .remote_addr = ntohl(remote_addr.sin_addr.s_addr),
                         .local_port = ntohs(local_addr.sin_port),
                         .remote_port = ntohs(remote_addr.sin_port)};
  // RM_PRINTF("flow: %u:%u->%u:%u\n", flow.remote_addr, flow.remote_port,
  //           flow.local_addr, flow.local_port);
  fd_to_flow.insert({new_fd, flow});

  // Should this flow be active or paused?
  if (active_fds.size() < max_active_flows) {
    // Less than the max number of flows are active, so make this one active.
    active_fds.insert(new_fd);
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds.insert(new_fd);
    // Pausing a flow means retting its RWND to 0 B.
    bpf_map_update_elem(flow_to_rwnd_fd, &flow, &zero, BPF_ANY);
  }

  RM_PRINTF("Successful 'accept' for FD=%d, got FD=%d\n", sockfd, new_fd);
  return new_fd;
}

// Get around C++ function name mangling.
extern "C" {
int close(int sockfd) {
  static int (*real_close)(int) = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == NULL) {
    RM_PRINTF("ERROR when querying dlsym for 'close': %s\n", dlerror());
    return -1;
  }
  int ret = real_close(sockfd);
  if (ret == -1) {
    RM_PRINTF("ERROR in real 'close'\n");
    return ret;
  }

  active_fds.erase(sockfd);
  paused_fds.erase(sockfd);
  // To get the flow struct for this FD, we must use visit() to look it up
  // in the concurrent_flat_map. Obviously, do this before removing the FD
  // from fd_to_flow.
  fd_to_flow.visit(sockfd, [](const auto &p) {
    bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);
  });
  fd_to_flow.erase(sockfd);

  RM_PRINTF("Successful 'close' for FD=%d\n", sockfd);
  return ret;
}
}
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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
boost::unordered::concurrent_flat_map<int, struct flow> fd_to_flow;

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
  // Variables used during each scheduling epoch.
  // Currently active flows, soon to be previously active.
  std::vector<int> prev_active_fds;
  // Length of prev_active_fds, which is the number of flows to be unpaused.
  size_t num_to_find = 0;
  // Previously paused flows that will be activated.
  std::vector<int> new_active_fds;
  // Preallocate suffient space.
  prev_active_fds.reserve(MAX_ACTIVE_FLOWS);
  new_active_fds.reserve(MAX_ACTIVE_FLOWS);

  printf("Thread started\n");

  while (true) {
    usleep(EPOCH_US);
    printf("Time to schedule\n");
    fflush(stdout);

    // If fewer than the max number of flows exist and they are all active, then
    // there is no need for scheduling.
    if (active_fds.size() < MAX_ACTIVE_FLOWS && paused_fds.size() == 0)
      continue;

    printf("Sufficient flows to schedule\n");

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
    for (const auto &fd : new_active_fds) {
      active_fds.insert(fd);
      paused_fds.erase(fd);
      fd_to_flow.visit(fd, [](const auto &p) {
        bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);
      });
      trigger_ack(fd);
    }

    // For each fo the previously active flows, add it to the paused set, remove
    // it from the active set, and install an RWND mapping to actually pause it.
    for (const auto &fd : prev_active_fds) {
      paused_fds.insert(fd);
      active_fds.erase(fd);
      fd_to_flow.visit(fd, [](const auto &p) {
        bpf_map_update_elem(flow_to_rwnd_fd, &p.second, &zero, BPF_ANY);
      });
    }

    prev_active_fds.clear();
    new_active_fds.clear();
  }
}

boost::thread t(thread_func);

// For some reason, C++ function name mangling does not prevent us from
// overriding accept().
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  printf("Intercepted call to 'accept'\n");

  static int (*real_accept)(int, struct sockaddr *, socklen_t *) =
      (int (*)(int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "accept");
  if (real_accept == NULL) {
    printf("Error in dlsym when querying for 'accept': %s\n", dlerror());
    return -1;
  }

  if (skel == NULL) {
    skel = ratemon_bpf__open();
    if (skel == NULL) {
      printf("ERROR when opening BPF skeleton\n");
    } else {
      int pinned_map_fd = bpf_obj_get(FLOW_TO_RWND_PIN_PATH);
      int err = bpf_map__reuse_fd(skel->maps.flow_to_rwnd, pinned_map_fd);
      if (err) {
        printf("ERROR when reusing map fd\n");
      } else {
        flow_to_rwnd = skel->maps.flow_to_rwnd;
        flow_to_rwnd_fd = pinned_map_fd;
        printf("Successfully reused map fd\n");
      }
    }
  }

  int new_fd = real_accept(sockfd, addr, addrlen);
  printf("accept for FD=%d got FD=%d\n", sockfd, new_fd);

  if (new_fd == -1) {
    return new_fd;
  }

  struct flow flow = {
      .local_addr = 0, .remote_addr = 0, .local_port = 0, .remote_port = 0};
  fd_to_flow.emplace(new_fd, flow);

  if (setsockopt(new_fd, SOL_TCP, TCP_CONGESTION, BPF_CUBIC,
                 strlen(BPF_CUBIC)) == -1) {
    printf("Error in setsockopt TCP_CONGESTION\n");
  }

  char retrieved_cca[32];
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(new_fd, SOL_TCP, TCP_CONGESTION, retrieved_cca,
                 &retrieved_cca_len) == -1) {
    printf("Error in getsockopt TCP_CONGESTION\n");
  }
  if (strcmp(retrieved_cca, BPF_CUBIC)) {
    printf("Error in setting CCA to %s! Actual CCA is: %s\n", BPF_CUBIC,
           retrieved_cca);
  }

  // Should this flow be active or paused?
  if (active_fds.size() < MAX_ACTIVE_FLOWS) {
    // Less than the max number of flows are active, so make this one active.
    active_fds.insert(new_fd);
  } else {
    // The max number of flows are active already, so pause this one.
    paused_fds.insert(new_fd);
    // Pausing a flow means retting its RWND to 0 B.
    bpf_map_update_elem(flow_to_rwnd_fd, &flow, &zero, BPF_ANY);
  }

  return new_fd;
}

// Get around C++ function name mangling.
extern "C" {
int close(int sockfd) {
  printf("Intercepted call to 'close' for FD=%d\n", sockfd);

  static int (*real_close)(int) = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == NULL) {
    fprintf(stderr, "Error in dlsym when querying for 'close': %s\n",
            dlerror());
    return -1;
  }

  int ret = real_close(sockfd);
  if (ret != -1) {
    active_fds.erase(sockfd);
    paused_fds.erase(sockfd);
    // To get the flow struct for this FD, we must use visit() to look it up in
    // the concurrent_flat_map. Obviously, do this before removing the FD from
    // fd_to_flow.
    fd_to_flow.visit(sockfd, [](const auto &p) {
      bpf_map_delete_elem(flow_to_rwnd_fd, &p.second);
    });
    fd_to_flow.erase(sockfd);
  }
  return ret;
}
}
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <linux/inet_diag.h>
#include <netinet/in.h> // structure for storing address information
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h> // for socket APIs
#include <unistd.h>

#include <boost/thread.hpp>
#include <boost/unordered/concurrent_flat_set.hpp>
#include <unordered_set>

const char *bpf_cubic = "bpf_cubic";

boost::unordered::concurrent_flat_set<int> sockfds_boost;

union tcp_cc_info placeholder_cc_info;
socklen_t placeholder_cc_info_length = (socklen_t)sizeof(placeholder_cc_info);

inline void trigger_ack(int fd) {
  // Do not store the output to check for errors since there is nothing we can
  // do.
  getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&placeholder_cc_info,
             &placeholder_cc_info_length);
}

void visit_fd(int fd) {
  printf("%d ", fd);
  trigger_ack(fd);
}

void thread_func() {
  printf("Thread started\n");

  while (true) {
    usleep(100000);
    printf("Thead looping...\n");

    printf("Visiting FDs: ");
    sockfds_boost.visit_all(visit_fd);
    printf("\n");

    fflush(stdout);
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

  int ret = real_accept(sockfd, addr, addrlen);
  printf("accept for FD=%d got FD=%d\n", sockfd, ret);

  if (ret == -1) {
    return ret;
  }

  sockfds_boost.insert(ret);

  if (setsockopt(ret, SOL_TCP, TCP_CONGESTION, bpf_cubic, strlen(bpf_cubic)) ==
      -1) {
    printf("Error in setsockopt TCP_CONGESTION\n");
  }

  char retrieved_cca[32];
  socklen_t retrieved_cca_len = sizeof(retrieved_cca);
  if (getsockopt(ret, SOL_TCP, TCP_CONGESTION, retrieved_cca,
                 &retrieved_cca_len) == -1) {
    printf("Error in getsockopt TCP_CONGESTION\n");
  }
  if (strcmp(retrieved_cca, bpf_cubic)) {
    printf("Error in setting CCA to bpf_cubic! Actual CCA is: %s\n",
           retrieved_cca);
  }

  sleep(5);
  return ret;
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
    sockfds_boost.erase(sockfd);
  }
  return ret;
}
}
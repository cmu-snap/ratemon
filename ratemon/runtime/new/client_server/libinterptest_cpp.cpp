#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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
#include <boost/unordered/concurrent_flat_set.hpp>
#include <unordered_set>

std::unordered_set<int> sockfds;

boost::unordered::concurrent_flat_set<int> sockfds_boost;

inline void trigger_ack(int fd) {
  if (fd == -1) {
    return;
  }
  union tcp_cc_info cc_info;
  socklen_t cc_info_length = (socklen_t)sizeof(cc_info);
  if (getsockopt(fd, SOL_TCP, TCP_CC_INFO, (void *)&cc_info, &cc_info_length) ==
      -1) {
    printf("Error in getsockopt TCP_CC_INFO\n");
  }
}

inline void visit_fd(int fd) {
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

  if (ret != -1) {
    sockfds.insert(ret);
    sockfds_boost.insert(ret);

    struct tcp_info info;
    socklen_t tcp_info_length = (socklen_t)sizeof(info);
    if (getsockopt(ret, SOL_TCP, TCP_INFO, (void *)&info, &tcp_info_length) ==
        0) {
      printf("TCP_INFO rtt=%u, rto=%u\n", info.tcpi_rtt, info.tcpi_rto);
    } else {
      printf("Error in getsockopt TCP_INFO\n");
    }

    const char *tcp_cong = "bpf_cubic";
    if (setsockopt(ret, SOL_TCP, TCP_CONGESTION, tcp_cong, strlen(tcp_cong)) ==
        0) {
      printf("Set TCP_CONGESTION to %s\n", tcp_cong);
    } else {
      printf("Error in setsockopt TCP_CONGESTION\n");
    }

    char retrieved_tcp_cong[32];
    socklen_t retrieved_tcp_cong_len = sizeof(retrieved_tcp_cong);
    if (getsockopt(ret, SOL_TCP, TCP_CONGESTION, retrieved_tcp_cong,
                   &retrieved_tcp_cong_len) == 0) {
      printf("Retrieved TCP_CONGESTION=%s\n", retrieved_tcp_cong);
    } else {
      printf("Error in getsockopt TCP_CONGESTION\n");
    }

    trigger_ack(ret);
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
    if (sockfds.find(sockfd) != sockfds.end()) {
      sockfds.erase(sockfd);
    }
    sockfds_boost.erase(sockfd);
  }

  return ret;
}
}
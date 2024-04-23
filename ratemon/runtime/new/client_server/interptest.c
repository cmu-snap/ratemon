#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>  // structure for storing address information
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>  // for socket APIs
#include <sys/types.h>

int storedSockFd = 0;

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  static int (*real_accept)(int, struct sockaddr *, socklen_t *) = NULL;
  if (!real_accept) {
    real_accept = dlsym(RTLD_NEXT, "accept");
  }

  int ret = real_accept(sockfd, addr, addrlen);
  if (ret != -1) {
    storedSockFd = ret;
  }

  printf("Intercepted call to 'accept', got FD=%d\n", storedSockFd);

  struct tcp_info info;
  int tcp_info_length = sizeof(info);
  if (getsockopt(ret, SOL_TCP, TCP_INFO, (void *)&info, &tcp_info_length) ==
      0) {
    printf("TCP_INFO rtt=%u, rto=%u\n", info.tcpi_rtt, info.tcpi_rto);
  } else {
    printf("Error in getsockopt TCP_INFO\n");
  }

  return ret;
}

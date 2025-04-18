#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h> // structure for storing address information
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h> // for socket APIs

int storedSockFd = 0;

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  printf("Intercepted call to 'accept'\n");

  static int (*real_accept)(int, struct sockaddr *, socklen_t *) = NULL;
  real_accept =
      (int (*)(int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "accept");
  if (real_accept == NULL) {
    fprintf(stderr, "Error in dlsym when querying for 'accept': %s\n",
            dlerror());
    return -1;
  }

  int ret = real_accept(sockfd, addr, addrlen);
  printf("accept for FD=%d got FD=%d\n", sockfd, ret);

  if (ret != -1) {
    struct tcp_info info;
    socklen_t tcp_info_length = (socklen_t)sizeof(info);
    if (getsockopt(ret, SOL_TCP, TCP_INFO, (void *)&info, &tcp_info_length) ==
        0) {
      printf("TCP_INFO rtt=%u, rto=%u\n", info.tcpi_rtt, info.tcpi_rto);
    } else {
      printf("Error in getsockopt TCP_INFO\n");
    }
  }

  return ret;
}

int close(int sockfd) {
  printf("Intercepted call to 'close' for FD=%d\n", sockfd);

  static int (*real_close)(int) = NULL;
  real_close = (int (*)(int))dlsym(RTLD_NEXT, "close");
  if (real_close == NULL) {
    fprintf(stderr, "Error in dlsym when querying for 'close': %s\n",
            dlerror());
    return -1;
  }

  return real_close(sockfd);
  ;
}

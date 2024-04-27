#include <netinet/in.h>  // structure for storing address information
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>  // for socket APIs
#include <sys/types.h>
#include <unistd.h>  // for close()

#include "common.h"

int main(int argc, char const* argv[]) {
  int sockD = socket(AF_INET, SOCK_STREAM, 0);
  if (sockD == -1) {
    printf("Error in creating socket\n");
    return 1;
  }

  struct sockaddr_in servAddr;
  servAddr.sin_family = AF_INET;
  servAddr.sin_port = htons(SERVER_PORT);  // use some unused port number
  servAddr.sin_addr.s_addr = INADDR_ANY;

  if (connect(sockD, (struct sockaddr*)&servAddr, sizeof(servAddr)) == -1) {
    printf("Error in connecting\n");
    return 1;
  }

  char strData[255];
  if (recv(sockD, strData, sizeof(strData), 0) == -1) {
    printf("Error in receiving\n");
    return 1;
  }

  printf("Message: %s\n", strData);
  printf("Client done receiving. Exiting...\n");

  // close server socket
  if (close(sockD) == -1) {
    printf("Error in closing socket\n");
    return 1;
  }
  return 0;
}

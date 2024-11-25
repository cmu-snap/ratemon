#include <netinet/in.h> // structure for storing address information
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h> // for socket APIs
#include <sys/types.h>
#include <unistd.h> // for close()

#include "common.h"

int main(int argc, char const *argv[]) {
  // create server socket similar to what was done in
  // client program
  int servSockD = socket(AF_INET, SOCK_STREAM, 0);
  if (servSockD == -1) {
    printf("Error in creating socket\n");
    return 1;
  }
  if (setsockopt(servSockD, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) ==
      -1) {
    printf("Error in setting SO_REUSEADDR\n");
    return 1;
  }

  // string store data to send to client
  const char serMsg[255] = "Message from the server to the "
                           "client \'Hello Client\' ";

  // define server address
  struct sockaddr_in servAddr;
  servAddr.sin_family = AF_INET;
  servAddr.sin_port = htons(SERVER_PORT);
  servAddr.sin_addr.s_addr = INADDR_ANY;

  // bind socket to the specified IP and port
  if (bind(servSockD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
    printf("Error in binding\n");
    return 1;
  }

  // listen for connections
  if (listen(servSockD, 1) == -1) {
    printf("Error in listening\n");
    return 1;
  }

  // integer to hold client socket.
  int clientSocket = accept(servSockD, NULL, NULL);
  if (clientSocket == -1) {
    printf("Error in accepting\n");
    return 1;
  }

  // send's messages to client socket
  if (send(clientSocket, serMsg, sizeof(serMsg), 0) == -1) {
    printf("Error in sending\n");
    return 1;
  }

  printf("Server done sending. Exiting...\n");

  if (close(clientSocket) == -1) {
    printf("Error in closing client socket\n");
    return 1;
  }
  // close server listen socket
  if (close(servSockD) == -1) {
    printf("Error in closing server socket\n");
    return 1;
  }
  return 0;
}

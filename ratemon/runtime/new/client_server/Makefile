CC = gcc
CFLAGS = -I. -g
DEPS = common.h
CLIENT_OBJ = client.o
SERVER_OBJ = server.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

client: $(CLIENT_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

server: $(SERVER_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

libinterptest: interptest.c
	$(CC) -shared -ldl -fPIC interptest.c -o libinterptest.so


all: client server libinterptest

clean:
	rm -f *.o *.so client server

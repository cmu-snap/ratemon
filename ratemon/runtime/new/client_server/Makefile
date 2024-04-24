OUTPUT = .output
CC = gcc
CXX = g++
CFLAGS = -g -std=c17 -Wall -Wextra
CXXFLAGS = -g -std=c++20 -Wall -Wextra
DEPS = common.h
APPS = client server
INTERPS = libinterptest libinterptest_cpp
BOOST_INCLUDE = /home/ccanel/src/boost_1_85_0/
BOOST_LIB = /home/ccanel/src/boost_1_85_0/stage/lib

.PHONY: all
all: $(APPS) $(INTERPS)

$(OUTPUT):
	mkdir -p $@

$(OUTPUT)/%.o: %.c $(DEPS) | $(OUTPUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(APPS): %: $(OUTPUT)/%.o | $(OUTPUT)
	$(CC) $(CFLAGS) -o $@ $^

$(OUTPUT)/%.so: %.c | $(OUTPUT)
	$(CC) $(CFLAGS) -shared -ldl -fPIC $^ -o $@

$(OUTPUT)/%_cpp.so: %_cpp.cpp | $(OUTPUT)
	$(CXX) $(CXXFLAGS) -shared -fPIC -I$(BOOST_INCLUDE) $^ -ldl -L${BOOST_LIB} -lboost_thread -o $@

$(INTERPS): %: $(OUTPUT)/%.so ;

interp_server : server $(OUTPUT)/libinterptest.so
	LD_PRELOAD=$(OUTPUT)/libinterptest.so ./server

interp_server_cpp : server $(OUTPUT)/libinterptest_cpp.so
	LD_LIBRARY_PATH=$(BOOST_LIB):${LD_LIBRARY_PATH} LD_PRELOAD=$(OUTPUT)/libinterptest_cpp.so ./server

run_server: server
	./server

run_client: client
	./client

.PHONY: clean
clean:
	rm -rfv $(OUTPUT) $(APPS)

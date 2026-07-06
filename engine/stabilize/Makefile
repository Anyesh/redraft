CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

test: test_stabilize test_batched
	./test_stabilize
	./test_batched

test_stabilize: test_stabilize.cpp stabilize.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_stabilize.cpp

test_batched: test_batched.cpp batched_stabilize.hpp stabilize.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_batched.cpp

clean:
	rm -f test_stabilize test_batched

.PHONY: test clean

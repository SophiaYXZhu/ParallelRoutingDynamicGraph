CXX      = mpic++
CXXFLAGS = -Wall -Wextra -O3 -std=c++20 -I. -fopenmp -Wno-free-nonheap-object

all: tests

dynamicgraph.o: dynamicgraph.cpp dynamicgraph.h
	$(CXX) $(CXXFLAGS) -c dynamicgraph.cpp

run_dynamic_graph.o: run_dynamic_graph.cpp dynamicgraph.h
	$(CXX) $(CXXFLAGS) -c run_dynamic_graph.cpp

run_dynamic_graph: dynamicgraph.o run_dynamic_graph.o
	$(CXX) $(CXXFLAGS) -o $@ dynamicgraph.o run_dynamic_graph.o

TEST_SRCS := $(shell find tests -name '*.cpp')
TEST_BINS := $(TEST_SRCS:.cpp=)

tests: $(TEST_BINS)

tests/%: tests/%.cpp dynamicgraph.o
	$(CXX) $(CXXFLAGS) -o $@ $< dynamicgraph.o

clean:
	rm -f *.o run_dynamic_graph
	rm -f $(TEST_BINS)
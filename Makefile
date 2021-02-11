CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
OBJS = test

all: $(OBJS)

clean:
	$(RM) $(OBJS)

.PHONY: all clean

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -o $@ -c $^
.cpp:
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
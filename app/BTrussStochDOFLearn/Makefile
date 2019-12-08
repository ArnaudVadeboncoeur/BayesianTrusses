SHELL = /bin/bash
CXX   = g++
LD    = g++

# includes
#INCLUDES  = -I$(ThreedTrussDef.hpp)
#INCLUDES += -I$(FEMClass.hpp)
INCLUDES += -I$(EIGEN_DIR)

# libraries
LIBS  = 

# common flags
CPPFLAGS  = -std=c++11
CPPFLAGS += $(INCLUDES)

# the final target
TARGET = compiled.exe

# find the sources and object files
CPPFILES = $(wildcard *.cpp)
CPPOBJS = $(CPPFILES:.cpp=.o)

#define sources and objects
SRCS = $(CPPFILES)
OBJS = $(CPPOBJS)

# for cleaning up
RM = -rm -f

#######################################
# the rules
#

# if nothing is given, make the target
all: $(TARGET)

# link the target
$(TARGET): $(OBJS) 
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS) 
	
# implicit rule for makin object files
.cpp.o:	
	$(CXX) $(CPPFLAGS) -c $<

clean:
	$(RM) $(OBJS) $(TARGET)
	
# implicit rule for makin object files
#.cpp.o:	
#	$(CXX) $(CPPFLAGS) $(OBJS) -c $(SRCS)
#
#clean:
#	$(RM) $(OBJS) $(TARGET)


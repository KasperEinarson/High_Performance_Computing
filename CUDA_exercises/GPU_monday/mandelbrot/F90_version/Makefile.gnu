TARGET	= mandelbrot
OBJS	= main.o mandel.o timestamp.o writepng.o

OPT	= -g -O3
#ISA	= -m32 
PARA	= -fopenmp

PNGWRITERPATH =	pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

CCC	= gcc
CXX	= g++
CXXFLAGS= -I $(PNGWRTIPATH)

F90C  	= gfortran
F90FLAGS= $(OPT) $(ISA) $(PARA) $(XOPT)
.SUFFIXES: .f90

LIBS	= -L$(PNGWRTLPATH) -lpngwriter -lpng -lc -lstdc++

all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(F90C) $(F90FLAGS) -o $@ $(OBJS) $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE) -f $(MAKEFILE_LIST)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@cd pngwriter/src && $(MAKE) -f $(MAKEFILE_LIST) clean
	@rm -f $(PNGWRITERLIB)
	@rm -f $(TARGET)
	@rm -f mandelbrot.png


%.o: %.f90
	$(F90C) $(F90FLAGS) -c $<

# dependencies
#
main.o : main.f90 
timestamp.o: timestamp.f90
writepng.o: writepng.cc writepng.h
mandel.o: mandel.f90

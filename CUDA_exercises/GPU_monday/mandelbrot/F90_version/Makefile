TARGET	= mandelbrot
OBJS	= main.o mandel.o timestamp.o 
COBJS	= writepng.o

OPT	= -g -fast
ISA	= 
PARA	= 

PNGWRITERPATH =	pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

CCC	= CC
CXX	= CC
CXXFLAGS= -I$(PNGWRTIPATH)

F90C  	= f90
F90FLAGS= $(OPT) $(ISA) $(PARA) $(XOPT)
.SUFFIXES: .f90

LIBS	= -L$(PNGWRTLPATH) -lpngwriter -lpng -lfsu

all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) $(COBJS)
	$(CXX) $(F90FLAGS) -o $@ $(OBJS) $(COBJS)  $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE)

clean:
	@/bin/rm -f $(OBJS) $(COBJS) core

realclean: clean
	@cd pngwriter/src && $(MAKE) clean
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

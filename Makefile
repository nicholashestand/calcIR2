src     = calcIR2.cpp
exes    = calcIR2.exe
CC      = g++
LIBS    = -lgmx_reader -lxdrfile -lm -lfftw3

all: ${exes}

${exes}: ${src} calcIR2.h
	$(CC) $(src) -o $(exes) $(LIBS) -std=c++11 -fmax-errors=10 -fopenmp -lpthread

clean:
	rm calcIR2.exe

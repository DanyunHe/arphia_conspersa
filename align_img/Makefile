include ../config.mk

lflags=`gsl-config --libs` -L/usr/local/lib
iflags=`gsl-config --cflags` -I../../shared -I../voro++_2d

objs=bi_interp.o bi_image.o 
src=$(patsubst %.o,%.cc,$(objs))
execs=bic_test align_test

all:
	$(MAKE) executables

executables: $(execs)

include Makefile.dep

depend:
	$(cxx) $(cflags) $(iflags) -MM $(src) >Makefile.dep

clean:
	rm -rf $(objs) $(execs)

%.o: %.cc
	$(cxx) $(cflags) $(iflags) -c $<

align_test: align_test.cc bi_interp.o bi_image.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff 

bic_test: bic_test.cc bi_interp.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff

.PHONY: clean depend executables

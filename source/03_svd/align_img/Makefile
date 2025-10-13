include ../config.mk

lflags=`gsl-config --libs` -L/usr/local/lib -L/opt/local/lib 
iflags=`gsl-config --cflags` -L/opt/homebrew/opt/libpng/lib -I/opt/homebrew/opt/libpng/include -L/opt/homebrew/opt/libtiff/lib -I/opt/homebrew/opt/libtiff/include

objs=bi_interp.o bi_image.o png_output.o
src=$(patsubst %.o,%.cc,$(objs))
execs=bic_test align_test alignall_test align_single

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

align_test: align_test.cc bi_interp.o bi_image.o png_output.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff -lpng

bic_test: bic_test.cc bi_interp.o png_output.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff -lpng

alignall_test: alignall_test.cc bi_interp.o bi_image.o png_output.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff -lpng

align_single: align_single.cc bi_interp.o bi_image.o png_output.o
	$(cxx) $(cflags) $(iflags) -o $@ $^ $(lflags) -ltiff -lpng

.PHONY: clean depend executables

# compiler choice
CC    = gcc

all: fastmodules

.PHONY : fastmodules

fastmodules:
	python revolver/setup.py build_ext --inplace
	mv fastmodules*.so revolver/.

clean:
	rm -f revolver/*.*o
	rm -f revolver/fastmodules.c
	rm -f revolver/fastmodules*.so
	rm -f revolver/*.pyc

# compiler choice
CC    = gcc

all: fastmodules

.PHONY : fastmodules

fastmodules:
	python revolver/c/setup.py build_ext --inplace
	mv fastmodules*.so revolver/c/.

clean:
	rm -f revolver/c/*.*o
	rm -f revolver/c/fastmodules.c
	rm -f revolver/c/fastmodules*.so
	rm -f revolver/c/*.pyc
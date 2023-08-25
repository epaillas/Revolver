# compiler choice
CC    = gcc

all: fastmodules c

.PHONY : fastmodules

c:
	make -C revolver/c all

fastmodules:
	python revolver/setup.py build_ext --inplace
	mv fastmodules*.so revolver/.

clean:
	rm -f revolver/*.*o
	rm -f revolver/fastmodules.c
	rm -f revolver/fastmodules*.so
	rm -f revolver/*.pyc
	rm -f revolver/c/*.o
	rm -f revolver/c/*.exe
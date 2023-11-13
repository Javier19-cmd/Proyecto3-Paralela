all: pgm.o	houghBase houghConstant houghGlobal houghShared

houghBase:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o houghBase

houghShared:	houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -o houghShared

houghConstant:	houghConstant.cu pgm.o
	nvcc houghConstant.cu pgm.o -o houghConstant

houghGlobal:	houghGlobal.cu pgm.o
	nvcc houghGlobal.cu pgm.o -o houghGlobal

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o

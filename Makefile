#  CPE 453
#  -------------------
CC= g++
LD= g++
CFLAGS= -g -Wall -c
LDFLAGS= -g -Wall  

ALL= RasterMain.o Util/Tga.o Util/Header.o NewMeshParser/BasicModel.o Util/RasterizeFuncs.o

all:	$(ALL) rasterizer

rasterizer:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o rasterizer 

RasterMain.o:	 RasterMain.cpp RasterMain.h 
	$(CC) $(CFLAGS) -o $@ $<

Util/Tga.o:	 Util/Tga.cpp Util/Tga.h Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Util/RasterizeFuncs.o:	 Util/RasterizeFuncs.cpp Util/RasterizeFuncs.h NewMeshParser/BasicModel.h NewMeshParser/Model.h Util/Tga.h
	$(CC) $(CFLAGS) -o $@ $<

Util/Header.o:	 Util/Header.cpp Util/Header.h 
	$(CC) $(CFLAGS) -o $@ $<

NewMeshParser/BasicModel.o:	 NewMeshParser/BasicModel.cpp NewMeshParser/BasicModel.h NewMeshParser/Model.h NewMeshParser/utils.h 
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

#  CPE 453
#  -------------------
CC= nvcc 
LD= nvcc 
CFLAGS= -O3 -arch=compute_20 -code=sm_20 -c -I "./glm" 
LDFLAGS= -O3 -arch=compute_20 -code=sm_20  -I "./glm" 

ALL= RasterMain.o Util/Tga.o Util/Header.o NewMeshParser/BasicModel.o Util/RasterizeFuncs.o Util/RasterizeHelpers.o

all:	$(ALL) rasterizer

rasterizer:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o rasterizer 

RasterMain.o:	 RasterMain.cpp RasterMain.h 
	$(CC) $(CFLAGS) -o $@ $<

Util/Tga.o:	 Util/Tga.cpp Util/Tga.h Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Util/RasterizeFuncs.o:	 Util/RasterizeFuncs.cu Util/RasterizeFuncs.h NewMeshParser/BasicModel.h NewMeshParser/Model.h Util/Tga.h
	$(CC) $(CFLAGS) -o $@ $<

Util/RasterizeHelpers.o:	 Util/RasterizeHelpers.cpp Util/RasterizeHelpers.h NewMeshParser/BasicModel.h NewMeshParser/Model.h Util/Tga.h
	$(CC) $(CFLAGS) -o $@ $<

Util/Header.o:	 Util/Header.cpp Util/Header.h 
	$(CC) $(CFLAGS) -o $@ $<

NewMeshParser/BasicModel.o:	 NewMeshParser/BasicModel.cpp NewMeshParser/BasicModel.h NewMeshParser/Model.h NewMeshParser/utils.h 
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

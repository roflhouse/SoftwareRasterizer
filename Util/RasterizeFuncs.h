/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include "../NewMeshParser/BasicModel.h"
#include "Tga.h"

typedef struct {
   int xl;
   int yl;
   int xr;
   int yr;
} BoundingBox;

typedef struct {
   int a;
   int b;
   int c;
} Triangle;

typedef struct {
   float x;
   float y;
   float z;
} Vertex;

int rasterize( BasicModel &mesh, Tga &file );
int rasterizeCUDA( BasicModel &mesh, Tga &file );
int boundingBoxHit( BoundingBox box, int ys, int xs );
Triangle *createTriangles( BasicModel mesh, BoundingBox *boundingBoxes );

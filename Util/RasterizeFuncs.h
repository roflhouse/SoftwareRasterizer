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
   //bottom left
   int xl;
   int yl;
   //top right
   int xr;
   int yr;
} BoundingBox;

typedef struct {
   float r;
   float g;
   float b;
} Color;

typedef struct {
   int a;
   int b;
   int c;
   Color color;
} Triangle;

typedef struct {
   int x;
   int y;
   int z;
} Vertex;


int rasterize( BasicModel &mesh, Tga &file );
int rasterize2( BasicModel &mesh, Tga &file );
int rasterizeCUDA( BasicModel &mesh, Tga &file );
Triangle *createTriangles( BasicModel &mesh, BoundingBox **boundingBoxes,
       Vertex *screenVerts);
void createBB( BoundingBox &box, const Triangle &triangle,  Vertex *verts );
Vertex *convertVertices( BasicModel &mesh, int width, int height );

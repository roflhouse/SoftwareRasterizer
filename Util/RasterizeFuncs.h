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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
} Triangle;

typedef struct {
   int x;
   int y;
   float z;
} Vertex;

typedef struct {
   float x;
   float y;
   float z;
} Normal;


int rasterize( BasicModel &mesh, Tga &file );
int rasterize2( BasicModel &mesh, Tga &file );
int rasterizeCUDA( BasicModel &mesh, Tga &file );
Triangle *createTriangles( BasicModel &mesh, BoundingBox **boundingBoxes,
       Vertex *screenVerts);
void createBB( BoundingBox &box, const Triangle &triangle,  Vertex *verts );
Vertex *convertVertices( BasicModel &mesh, int width, int height );
Color calcLighting( Normal *normals, BasicModel &mesh, int v, Normal l );
Normal normalize( Normal n );
float dot( Normal n1, Normal n2 );
Normal *createNormals( BasicModel &mesh, int n );
Color *createColors( BasicModel &mesh, Normal *normals, Normal light );

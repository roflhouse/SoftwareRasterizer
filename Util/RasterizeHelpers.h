/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef RASTERIZEHELPERS_H
#define RASTERIZEHELPERS_H
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include "../NewMeshParser/BasicModel.h"
#include "Tga.h"
#include "RasterizeFuncs.h"
#include "Types.h"

Triangle *createTriangles( BasicModel &mesh, BoundingBox **boundingBoxes,
       Vertex *screenVerts);
void createBB( BoundingBox &box, const Triangle &triangle,  Vertex *verts );
Vertex *convertVertices( BasicModel &mesh, int width, int height );
Color calcLighting( Normal *normals, BasicModel &mesh, int v, Normal l );
Normal normalize( Normal n );
float dot( Normal n1, Normal n2 );
Normal *createNormals( BasicModel &mesh, int n );
Color *createColors( BasicModel &mesh, Normal *normals, Normal light );
#endif

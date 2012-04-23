/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef RASTERIZEFUNCS_H
#define RASTERIZEFUNCS_H

#define TILE_WIDTH 16

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include "../NewMeshParser/BasicModel.h"
#include "Tga.h"
#include "Types.h"

int rasterize( BasicModel &mesh, Tga &file );

#endif

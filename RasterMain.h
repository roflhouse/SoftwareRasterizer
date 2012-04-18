/**
 *  CPE 2012
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
#include <sstream>
#include <iostream>

#include "Util/Tga.h"
#include "Util/RasterizeFuncs.h"
#include "NewMeshParser/BasicModel.h"

char *parseCommandLine( int argc, char *argv[] );

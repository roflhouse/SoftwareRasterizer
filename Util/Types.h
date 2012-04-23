#ifndef TYPES_H
#define TYPES_H
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
typedef struct {

   float r;
   float g;
   float b;
} pixel;
#endif

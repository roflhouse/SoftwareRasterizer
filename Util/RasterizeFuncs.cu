/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "RasterizeFuncs.h"
#include "RasterizeHelpers.h"
#include "../NewMeshParser/utils.h"
#include <vector>
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#include "cutil.h"
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }


__global__ void rasterizeCUDA_Dev( int width, int height, int num_tri, pixel *data, Vertex *vertices,
      Triangle *triangles, BoundingBox *boundingBoxes, Color *colors, double *depth, unsigned int *mutex )
{
   int triIndex = blockIdx.x * gridDim.x + blockIdx.y;
   if( triIndex >= num_tri )
      return;
   pixel pix;
   int width_bb;
   int height_bb;

   __shared__ Triangle triangle;
   __shared__ BoundingBox box;
   //Vertices for this triangle
   __shared__ Vertex a;
   __shared__ Vertex b;
   __shared__ Vertex c;
   //Colors
   __shared__ Color a_c;
   __shared__ Color b_c;
   __shared__ Color c_c;

   //Fill out all triangle shared data;
   if( threadIdx.x == 0 && threadIdx.y == 0 )
      triangle = triangles[triIndex];
   else if (threadIdx.x == 1 && threadIdx.y == 0 )
      box = boundingBoxes[triIndex];

   __syncthreads();
   //Fill out all Vertex shared data;
   if( threadIdx.y != 0 ){
      //do nothing
   }
   else if( threadIdx.x == 0 )
      a = vertices[triangle.a];
   else if( threadIdx.x == 1 )
      b = vertices[triangle.b];
   else if( threadIdx.x == 2 )
      c = vertices[triangle.c];
   else if( threadIdx.x == 3 )
      a_c = colors[triangle.a];
   else if( threadIdx.x == 4 )
      b_c = colors[triangle.b];
   else if( threadIdx.x == 5 )
      c_c = colors[triangle.c];
   __syncthreads();

   width_bb = box.xr - box.xl;
   height_bb = box.yr - box.yl;
   int x_tile = width_bb / TILE_WIDTH;
   int y_tile = height_bb / TILE_WIDTH;
   if( width_bb % TILE_WIDTH )
      x_tile++;
   if( height_bb % TILE_WIDTH )
      y_tile++;

   for( int m = 0; m < x_tile; m++ )
   {
      int j = TILE_WIDTH * m + threadIdx.x + box.xl;
      if( j >= width )
      {
         break;
      }
      if( j < 0 )
      {
         continue;
      }
      for( int n = 0; n < y_tile; n++ )
      {
         int i = TILE_WIDTH * n + threadIdx.y + box.yl;
         if( i >= height )
         {
            break;
         }
         if( i < 0 )
         {
            continue;
         }

         if( i < box.yl || i > box.yr || j > box.xr || j < box.xl)
         {
            continue;
         }
         //These are alot of shared mem accesses but less registers. Could use register might be faster
         double beta = (double)((a.x-c.x)*(i-c.y) - (j-c.x)*(a.y-c.y))
            /(double)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
         double gamma = (double)((b.x-a.x)*(i-a.y) - (j-a.x)*(b.y-a.y))/
            (double)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
         double alpha;
         if( beta+gamma <= 1.010 && beta >=-0.010 && gamma >= -0.010 ){
            alpha = 1- beta -gamma;
         }
         else
         {
            continue;
         }

         double depthTemp = a.z * alpha + b.z * beta + c.z *gamma;
         pix.r = a_c.r*alpha + b_c.r*beta + c_c.r*gamma;
         pix.g = a_c.g*alpha + b_c.g*beta + c_c.g*gamma;
         pix.b = a_c.b*alpha + b_c.b*beta + c_c.b*gamma;
         if( depthTemp > depth[i*width + j] )
         {
            while( atomicInc( &(mutex[i*width + j]), 1 ) ){};
            if( depthTemp > depth[i*width + j] )
            {
               depth[i*width + j] = depthTemp;
               data[i*width + j] = pix;
            }
            atomicDec( &(mutex[i*width + j]), 0 );
         }
      }
   }
}
int rasterize( BasicModel &mesh, Tga &file )
{
   Normal light;
   light.x = 3;
   light.y = 3;
   light.z = 3;
   pixel **data = file.getBuffer();
   int width = file.getWidth();
   int height = file.getHeight();
   double *depth = (double*) malloc(sizeof(double) * width * height );
   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         depth[width * i +j] = -100000;
   unsigned int tris = mesh.Triangles.size();

   //Converts mesh verts to screenspace
   Vertex *vertices = convertVertices( mesh, width, height );

   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );
   Normal *normals = createNormals( mesh, (int)mesh.Vertices.size() );
   Color *colors = createColors( mesh, normals, light );
   printf("Number: %d %d\n", (int )mesh.Vertices.size(), (int)tris);

   Vertex *d_vert;
   Triangle *d_tri;
   BoundingBox *d_box;
   Color *d_color;
   double *d_depth;
   unsigned int *d_mutex;
   pixel *d_data;
   pixel *temp;

   temp = (pixel *) malloc( sizeof(pixel) * width * height );
   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         temp[i*width + j] = data[i][j];

   //malloc all the data;
   CUDASAFECALL(cudaMalloc( (void **)&d_depth, sizeof(double) * width * height ));
   CUDASAFECALL(cudaMalloc( (void **)&d_vert, sizeof(Vertex) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_tri, sizeof(Triangle) * tris )  );
   CUDASAFECALL(cudaMalloc( (void **)&d_box, sizeof(BoundingBox) * tris ));
   CUDASAFECALL(cudaMalloc( (void **)&d_color, sizeof(Color) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_mutex, sizeof(unsigned int) * width * height ));
   CUDASAFECALL(cudaMalloc( (void **)&d_data, sizeof(pixel) * width * height ) );

   //Zero the mutexes
   CUDASAFECALL(cudaMemset( d_mutex, 0, width * height * sizeof(unsigned int)));

   //copy over all the data;
   CUDASAFECALL(cudaMemcpy( d_data, temp, sizeof(pixel) * width*height, cudaMemcpyHostToDevice));
   CUDASAFECALL(cudaMemcpy( d_depth, depth, sizeof(double) * width * height, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_vert, vertices, sizeof(Vertex) * mesh.Vertices.size(), cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_tri, triangles, sizeof(Triangle) *tris, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_box, boundingBoxes, sizeof(BoundingBox) * tris, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_color, colors, sizeof(Color) * mesh.Vertices.size(), cudaMemcpyHostToDevice ));

   unsigned int x;
   x = sqrt( tris );
   if ( x < sqrt( tris ) )
      x++;
   dim3 dimBlock( TILE_WIDTH, TILE_WIDTH );
   dim3 dimGrid( x, x );

   printf("Starting Kernel\n");
   rasterizeCUDA_Dev<<< dimGrid, dimBlock >>>(width, height, tris, d_data, d_vert,
         d_tri, d_box, d_color, d_depth, d_mutex );
   printf("Ending Kernel\n");
   CUDAERRORCHECK();

   CUDASAFECALL(cudaMemcpy( temp, d_data, sizeof(pixel) * width * height, cudaMemcpyDeviceToHost ));
   cudaFree( d_vert );
   cudaFree( d_tri );
   cudaFree( d_box );
   cudaFree( d_color );
   cudaFree( d_mutex );
   cudaFree( d_data );
   cudaFree( d_depth );

   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         data[i][j] = temp[i*width + j];
   free(temp);

   return 0;
}

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


__global__ void rasterizeCUDA_Dev( int width, int height, int offx, int offy, int num_tri, pixel *data,
      Vertex *vertices, Triangle *triangles, BoundingBox *boundingBoxes, Color *colors,
      float *depth, unsigned int *mutex )
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

   if( threadIdx.x == 0 && threadIdx.y == 1 )
      b_c = colors[triangle.b];
   else if( threadIdx.x == 1 && threadIdx.y == 1 )
      c_c = colors[triangle.c];
   else if( threadIdx.y != 0 ){
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
   __syncthreads();

   width_bb = box.xr - box.xl;
   height_bb = box.yr - box.yl;
   int x_tile = width_bb / TILE_WIDTH;
   int y_tile = height_bb / TILE_WIDTH;
   if( width_bb % TILE_WIDTH )
      x_tile++;
   if( height_bb % TILE_WIDTH )
      y_tile++;

   for( int n = 0; n < y_tile; n++ )
   {
      int i = TILE_WIDTH * n + threadIdx.y + box.yl;
      if( i >= height || i + offy >= height )
      {
         break;
      }
      if( i < 0 || i + offy < 0 )
      {
         continue;
      }
      for( int m = 0; m < x_tile; m++ )
      {
         int j = TILE_WIDTH * m + threadIdx.x + box.xl;
         if( j >= width || j + offx >= width )
         {
            break;
         }
         if( j < 0  || j + offx < 0)
         {
            continue;
         }

         if( i < box.yl || i > box.yr || j > box.xr || j < box.xl)
         {
            continue;
         }
         //These are alot of shared mem accesses but less registers. Could use register might be faster
         float beta = (float)((a.x-c.x)*(i-c.y) - (j-c.x)*(a.y-c.y))
            /(float)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
         float gamma = (float)((b.x-a.x)*(i-a.y) - (j-a.x)*(b.y-a.y))/
            (float)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
         float alpha;
         if( beta+gamma <= 1.010 && beta >=-0.010 && gamma >= -0.010 )
            alpha = 1- beta -gamma;
         else
            continue;

         float depthTemp = a.z * alpha + b.z * beta + c.z *gamma;
         pix.r = a_c.r*alpha + b_c.r*beta + c_c.r*gamma;
         pix.g = a_c.g*alpha + b_c.g*beta + c_c.g*gamma;
         pix.b = a_c.b*alpha + b_c.b*beta + c_c.b*gamma;
         if( depthTemp > depth[(offy+i)*width + j+offx] )
         {
            while( atomicInc( &(mutex[(i+offy)*width + j + offx]), 1 ) ){};
            if( depthTemp > depth[(i+offy)*width + j+offx] )
            {
               depth[(i+offy)*width + j + offx ] = depthTemp;
               data[(i+offy)*width + j + offx] = pix;
            }
            atomicDec( &(mutex[(offy+i)*width + j +offx]), 0 );
         }
      }
   }
}
__global__ void initData( pixel *data, float *depth, int width, int height ){
   int i = blockIdx.x * TILE_WIDTH + threadIdx.x;
   int j = blockIdx.y * TILE_WIDTH + threadIdx.y;

   if( i < width && j < height )
   {
      data[j*width + i].r = 0;
      data[j*width + i].g = 1;
      data[j*width + i].b = 0;
      depth[j*width + i] = -100000;
   }
}
int rasterize( BasicModel &mesh, Tga &file )
{
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEvent_t start1, stop1;
   cudaEventCreate(&start1);
   cudaEventCreate(&stop1);

   cudaEventRecord(start, 0);

   Normal light;
   light.x = 3;
   light.y = 3;
   light.z = 3;
   pixel *data = file.getBuffer();
   int width = file.getWidth();
   int height = file.getHeight();
   unsigned int tris = mesh.Triangles.size();

   //Converts mesh verts to screenspace
   Vertex *vertices = convertVertices( mesh, width, height );

   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );
   Normal *normals = createNormals( mesh, (int)mesh.Vertices.size() );
   Color *colors = createColors( mesh, normals, light );

   Vertex *d_vert;
   Triangle *d_tri;
   BoundingBox *d_box;
   Color *d_color;
   float *d_depth;
   unsigned int *d_mutex;
   pixel *d_data;

   CUDASAFECALL(cudaMalloc( (void **)&d_depth, sizeof(float) * width * height ));
   CUDASAFECALL(cudaMalloc( (void **)&d_vert, sizeof(Vertex) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_tri, sizeof(Triangle) * tris )  );
   CUDASAFECALL(cudaMalloc( (void **)&d_box, sizeof(BoundingBox) * tris ));
   CUDASAFECALL(cudaMalloc( (void **)&d_data, sizeof(pixel) * width * height ) );
   CUDASAFECALL(cudaMalloc( (void **)&d_color, sizeof(Color) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_mutex, sizeof(unsigned int) * width * height ));

   int w = width / TILE_WIDTH;
   if( w < (float)width / (float)TILE_WIDTH )
      w++;
   int h = height / TILE_WIDTH;
   if( h < (float)height / (float)TILE_WIDTH )
      h++;
   dim3 dimBlock1( TILE_WIDTH, TILE_WIDTH );
   dim3 dimGrid1( w, h );
   initData<<<dimGrid1, dimBlock1>>>( d_data, d_depth, width, height );

   CUDASAFECALL(cudaMemcpyAsync( d_vert, vertices, sizeof(Vertex) * mesh.Vertices.size(), cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpyAsync( d_tri, triangles, sizeof(Triangle) *tris, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpyAsync( d_box, boundingBoxes, sizeof(BoundingBox) * tris, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpyAsync( d_color, colors, sizeof(Color) * mesh.Vertices.size(), cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemsetAsync( d_mutex, 0, width * height * sizeof(unsigned int)));

   cudaDeviceSynchronize();

   unsigned int x;
   x = sqrt( tris );
   if ( x < sqrt( tris ) )
      x++;
   dim3 dimBlock( TILE_WIDTH, TILE_WIDTH );
   dim3 dimGrid( x, x );

   cudaEventRecord(start1, 0);
   printf("Starting Kernel\n");
   for( int i = 0; i < 5; i++ )
   {
      for( int j = 0; j < 5; j++ )
      {
         rasterizeCUDA_Dev<<< dimGrid, dimBlock >>>(width, height,width/5 * i, height/5 * j, tris, d_data, d_vert,
               d_tri, d_box, d_color, d_depth, d_mutex );
      }
   }
   CUDAERRORCHECK();

   cudaEventRecord(stop1, 0);

   CUDASAFECALL(cudaMemcpy( data, d_data, sizeof(pixel) * width * height, cudaMemcpyDeviceToHost ));
   printf("Ending Kernel\n");
   cudaFree( d_vert );
   cudaFree( d_tri );
   cudaFree( d_box );
   cudaFree( d_color );
   cudaFree( d_mutex );
   cudaFree( d_data );
   cudaFree( d_depth );

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);

   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   printf("Cuda Time: %f\n", elapsedTime);

   cudaEventElapsedTime(&elapsedTime, start1, stop1);
   printf("Cuda Time (no memcpy): %f\n", elapsedTime);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaEventDestroy(start1);
   cudaEventDestroy(stop1);

   return 0;
}

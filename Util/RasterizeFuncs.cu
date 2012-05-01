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
#define CONST_MEM_PIXELS 5120

__device__ __constant__ pixel constPixels[CONST_MEM_PIXELS];

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
   if( threadIdx.x == 0)
      triangle = triangles[triIndex];
   else if (threadIdx.x == 1)
      box = boundingBoxes[triIndex];

   __syncthreads();
   //Fill out all Vertex shared data;
   if( threadIdx.x == 0 )
      a = vertices[triangle.a];
   else if( threadIdx.x == 1 )
      b = vertices[triangle.b];
   else if( threadIdx.x == 2 )
      c = vertices[triangle.c];
   else if( threadIdx.x == 3 )
      a_c = colors[triangle.a];
   else if( threadIdx.x == 4)
      b_c = colors[triangle.b];
   else if( threadIdx.x == 5)
      c_c = colors[triangle.c];
   __syncthreads();

   width_bb = box.xr - box.xl;
   height_bb = box.yr - box.yl;
   int pxIdx, i, j;
   int bb_size = width_bb * height_bb;
   int loop = bb_size / TILE_WIDTH;
   if( loop < ((float)bb_size) / ((float)TILE_WIDTH))
      loop++;

   for( int n = 0; n < loop; n++ )
   {
      pxIdx = n * TILE_WIDTH + threadIdx.x;

      if( pxIdx >= bb_size)
         return;

      i = pxIdx / width_bb + box.yl;
      j = pxIdx % width_bb + box.xl;

      if((i < 0 && i >= height) || (j < 0 && j >= width) || (i + offy >= height) || (j + offx >= width))
         continue;

      //These are alot of shared mem accesses but less registers. Could use register might be faster
      float beta = (float)((a.x-c.x)*(i-c.y) - (j-c.x)*(a.y-c.y))
         /(float)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
      float gamma = (float)((b.x-a.x)*(i-a.y) - (j-a.x)*(b.y-a.y))/
         (float)((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y));
      float alpha;
      if( beta+gamma <= 1.01 && beta >=-0.01 && gamma >= -0.01 )
         alpha = 1- beta -gamma;
      else
         continue;

      float depthTemp = a.z * alpha + b.z * beta + c.z *gamma;
      pix.r = a_c.r*alpha + b_c.r*beta + c_c.r*gamma;
      pix.g = a_c.g*alpha + b_c.g*beta + c_c.g*gamma;
      pix.b = a_c.b*alpha + b_c.b*beta + c_c.b*gamma;
      /*for( int h = 0; h < TILE_WIDTH; h++ )
        {
        if( threadIdx.x == h )
        {
        while( !atomicInc( &(mutex[(i+offy)*width + j + offx]), 1) ) {};
        }
        __threadfence();
        __syncthreads();
        }
       */
      while( !atomicInc( &(mutex[(i+offy)*width +j + offx]), 1) ) {};
      if( depthTemp > depth[(i+offy)*width + j+offx] )
      {
         depth[(i+offy)*width + j + offx ] = depthTemp;
         data[(i+offy)*width + j + offx] = pix;
      }
      atomicDec( &(mutex[(i+offy)*width + j +offx]), 0 );
   }
}
__global__ void initData( pixel *data, float *depth, int width, int height ){
   int i = blockIdx.x * INIT_WIDTH + threadIdx.x;
   int j = blockIdx.y * INIT_WIDTH + threadIdx.y;

   if( i < width && j < height )
   {
      data[j*width + i].r = 0;
      data[j*width + i].g = 1;
      data[j*width + i].b = 0;
      depth[j*width + i] = -100000;
   }
}
__global__ void blurHor( pixel *output, int width, int height )
{
   float weight[5];
   /*weight[0] = 0.225585938;
     weight[1] = 0.193359375;
     weight[2] = 0.120849609;
     weight[3] = 0.053710938;
     weight[4] = 0.016113281;
    */

   weight[0] = 0.2270270270;
   weight[1] = 0.1945945946;
   weight[2] = 0.1216216216;
   weight[3] = 0.0540540541;
   weight[4] = 0.0162162162;
   pixel outTemp;

   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j >= width || i >= height)
      return;

   int outIdx = j*height + i;
   pixel flow = constPixels[j];
   outTemp.r = flow.r * weight[0];
   outTemp.g = flow.g * weight[0];
   outTemp.b = flow.b * weight[0];
   for( int k = 1; k < 5; k++ )
   {
      int posIndex = j +k;
      int negIndex = j - k;
      if( posIndex >= width )
         posIndex = width-1;
      if( negIndex < 0 )
         negIndex = 0;

      flow = constPixels[posIndex];
      outTemp.r += flow.r * weight[k];
      outTemp.g += flow.g * weight[k];
      outTemp.b += flow.b * weight[k];

      flow = constPixels[negIndex];
      outTemp.r += flow.r * weight[k];
      outTemp.g += flow.g * weight[k];
      outTemp.b += flow.b * weight[k];
   }
   if( outTemp.r > 1 )
      outTemp.r = 1;
   if( outTemp.g > 1 )
      outTemp.g = 1;
   if( outTemp.b > 1 )
      outTemp.b = 1;
   //output[outIdx] = outTemp;
}
__global__ void blurVer( pixel *data, pixel *output, int width, int height )
{
   double weight[5];
   weight[0] = 0.2270270270;
   weight[1] = 0.1945945946;
   weight[2] = 0.1216216216;
   weight[3] = 0.0540540541;
   weight[4] = 0.0162162162;

   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   if (j >= width || i >= height)
      return;
   /*   for( int i = 0; i < height; i++ )
        {
        for( int j = 0; j < width; j++ )
        {*/
   int index = i*width + j;
   pixel outTemp;
   outTemp.r = data[index].r * weight[0];
   outTemp.g = data[index].g * weight[0];
   outTemp.b = data[index].b * weight[0];
   for( int k = 1; k < 5; k++ )
   {
      int posIndex = i +k;
      int negIndex = i - k;
      if( posIndex >= height )
         continue;//posIndex = height-1;
      if( negIndex < 0 )
         continue;//negIndex = 0;
      posIndex = posIndex *width + j;
      negIndex = negIndex *width + j;
      outTemp.r += data[posIndex].r * weight[k];
      outTemp.r += data[negIndex].r * weight[k];

      outTemp.g += data[posIndex].g * weight[k];
      outTemp.g += data[negIndex].g * weight[k];

      outTemp.b += data[posIndex].b * weight[k];
      outTemp.b += data[negIndex].b * weight[k];
   }
   if( outTemp.r > 1 )
      outTemp.r = 1;
   if( outTemp.g > 1 )
      outTemp.g = 1;
   if( outTemp.b > 1 )
      outTemp.b = 1;
   output[index] = outTemp;
   //      }
   //   }
}
int rasterize( BasicModel &mesh, Tga &file )
{
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEvent_t start1, stop1;
   cudaEventCreate(&start1);
   cudaEventCreate(&stop1);
   cudaEvent_t start2, stop2;
   cudaEventCreate(&start2);
   cudaEventCreate(&stop2);

   cudaEventRecord(start, 0);

   Normal light;
   light.x = 3;
   light.y = 3;
   light.z = 3;
   int width = file.getWidth();
   int height = file.getHeight();
   //   pixel *tempBuffer = (pixel *) malloc(sizeof(pixel) * width *height );
   pixel *data = file.getBuffer();
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
   pixel *d_buff;

   CUDASAFECALL(cudaMalloc( (void **)&d_depth, sizeof(float) * width * height ));
   CUDASAFECALL(cudaMalloc( (void **)&d_vert, sizeof(Vertex) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_tri, sizeof(Triangle) * tris )  );
   CUDASAFECALL(cudaMalloc( (void **)&d_box, sizeof(BoundingBox) * tris ));
   CUDASAFECALL(cudaMalloc( (void **)&d_data, sizeof(pixel) * width * height ) );
   CUDASAFECALL(cudaMalloc( (void **)&d_color, sizeof(Color) * mesh.Vertices.size() ));
   CUDASAFECALL(cudaMalloc( (void **)&d_mutex, sizeof(unsigned int) * width * height ));

   int w = width / INIT_WIDTH;
   if( w < (float)width / (float)INIT_WIDTH )
      w++;
   int h = height / INIT_WIDTH;
   if( h < (float)height / (float)INIT_WIDTH )
      h++;
   dim3 dimBlock1( INIT_WIDTH, INIT_WIDTH );
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
   dim3 dimBlock( TILE_WIDTH );
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
   /*dim3 dimBlock2(INIT_WIDTH, INIT_WIDTH);
   dim3 dimGrid2((height / INIT_WIDTH) + 1, (width / INIT_WIDTH) + 1);
   */
   dim3 dimBlock2( 256 );
   dim3 dimGrid2( 2000/256 +1 );

   CUDA_SAFE_CALL(cudaMalloc((void **) &d_buff, sizeof(pixel) * width * height));

   cudaEventRecord(start2, 0);
   for( int i = 0; i < 100; i++ )
   {
      for ( unsigned int j = 0; j < height; j++) {
         CUDASAFECALL(cudaMemcpyToSymbol( constPixels, d_data +(width * j ),
                  sizeof(pixel) * width, 0, cudaMemcpyDeviceToDevice ));
         blurHor<<<dimGrid2, dimBlock2>>>( d_buff, width, height );
      }

      for ( unsigned int j = 0; j < width; j++) {
         CUDASAFECALL(cudaMemcpyToSymbol( constPixels, d_buff + (height * j),
                  sizeof(pixel) * height, 0, cudaMemcpyDeviceToDevice ));
         blurHor<<<dimGrid2, dimBlock2>>>( d_data, height, width );
      }
   }
   cudaEventRecord(stop2, 0);
   CUDASAFECALL(cudaMemcpy( data, d_data, sizeof(pixel) * width * height, cudaMemcpyDeviceToHost ));


   printf("Ending Kernel\n");
   cudaFree( d_data );
   cudaFree( d_vert );
   cudaFree( d_tri );
   cudaFree( d_box );
   cudaFree( d_color );
   cudaFree( d_mutex );
   cudaFree( d_depth );

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);

   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, start, stop);
   printf("Cuda Time: %f\n", elapsedTime);

   cudaEventElapsedTime(&elapsedTime, start1, stop1);
   printf("Rasterize Time: %f\n", elapsedTime);

   cudaEventElapsedTime(&elapsedTime, start2, stop2);
   printf("Blur Time: %f\n", elapsedTime);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaEventDestroy(start1);
   cudaEventDestroy(stop1);
   cudaEventDestroy(start2);
   cudaEventDestroy(stop2);

   return 0;
}

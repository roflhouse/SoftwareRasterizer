/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "RasterizeFuncs.h"
#include <vector>

int rasterize( BasicModel &mesh, Tga &file )
{
   Tga::pixel **data = file.getBuffer();
   int width = file.getWidth();
   int height = file.getHeight();
   float depth[height][width];
   int tris = mesh.Trangles.size();
   Vertex *vertices;
   BoundingBox *boundingBoxes;
   Triangles *triangles = createTriangles( mesh, &boundingBoxes );

   for( int i = 0; i < height; ++i )
   {
      for( int j = 0; j < width; ++j )
      {
         depth[i][j] = -100000;
         for( int k = 0; k < tris; ++k )
         {
            if( boundingHitBox( boundingBoxes[k], i, j ) )
            {
               float xa = vertices(triangles[k].a).x;
               float ya = vertices(triangles[k].a).y;
               float xc = vertices(triangles[k].c).x;
               float yc = vertices(triangles[k].c).y;
               float yb = vertices(triangles[k].b).y;
               float xb = vertices(triangles[k].b).x;
               float beta = ((xa-xc)*(i-yc) - (j-xc)*(ya-yc))/((xc-xa)*(yb-ya) - (xb-xa)*(yc-ya));
               float gamma = ((xb-xa)*(i-ya) - (j-xa)*(yb-ya))/((xb-xa)*(yc-ya) - (xc-xa)*(yb-ya));
               float alpha;
               if( beta <= 1 && gamma <= 1 && beta+gamma <= 1 ){
                  alpha = 1- beta -gamma;
               }
               else
                  continue;

               float depthTemp = alpha*verices(triangles[k].a).z + beta*verices(triangles[k].b).z + gamma*verices(triangles[k].c).z;
               if( depthTemp < depth[i][j] || depth[i][j] == -100000 )
               {
                  data[i][j].r = alpha*verices(triangles[k].a).color.r + beta*verices(triangle[k].b).color.r + gamma*vertices(triangles[k].c).color.r;
                  data[i][j].g = alpha*verices(triangles[k].a).color.g + beta*verices(triangle[k].b).color.g + gamma*vertices(triangles[k].c).color.g;
                  data[i][j].b = alpha*verices(triangles[k].a).color.b + beta*verices(triangle[k].b).color.b + gamma*vertices(triangles[k].c).color.b;
                  depth[i][j] = depthTemp;
               }
            }
         }
      }
   }
   return 0;
}
Triangle *createTriangles( BasicModel &mesh, BoundingBox *boundingBoxes )
{
}
int boundingBoxHit( BoundingBox box, int ys, int xs )
{
}

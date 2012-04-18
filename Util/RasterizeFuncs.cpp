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
   int tris = mesh.Triangles.size();
   Vertex *vertices = convertVertices( mesh, width, height );
   printf("w: %d %d %d\n", vertices[0].x, vertices[0].y, vertices[0].z );
   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );

   for( int i = 0; i < height; ++i )
   {
      for( int j = 0; j < width; ++j )
      {
         depth[i][j] = 100000;
         for( int k = 0; k < tris; ++k )
         {
            if( boundingBoxes[k].xl <= j && boundingBoxes[k].yl <= i && boundingBoxes[k].xr >= j && boundingBoxes[k].yr >= i )
            {
               float xa = vertices[triangles[k].a].x;
               float ya = vertices[triangles[k].a].y;
               float xc = vertices[triangles[k].c].x;
               float yc = vertices[triangles[k].c].y;
               float yb = vertices[triangles[k].b].y;
               float xb = vertices[triangles[k].b].x;
               float beta = ((xa-xc)*(i-yc) - (j-xc)*(ya-yc))/((xc-xa)*(yb-ya) - (xb-xa)*(yc-ya));
               float gamma = ((xb-xa)*(i-ya) - (j-xa)*(yb-ya))/((xb-xa)*(yc-ya) - (xc-xa)*(yb-ya));
               float alpha;
               if( beta <= 1 && gamma <= 1 && beta+gamma <= 1 ){
                  alpha = 1- beta -gamma;
               }
               else
                  continue;

               float depthTemp = alpha*vertices[triangles[k].a].z + beta*vertices[triangles[k].b].z + gamma*vertices[triangles[k].c].z;
               if( depthTemp < depth[i][j] )
               {
                  data[i][j].r =  alpha*triangles[k].color.r + beta*triangles[k].color.r
                     + gamma*triangles[k].color.r;
                  data[i][j].g = alpha*triangles[k].color.g + beta*triangles[k].color.g
                     + gamma*triangles[k].color.g;
                  data[i][j].b = alpha*triangles[k].color.b + beta*triangles[k].color.b
                     + gamma*triangles[k].color.b;
                  depth[i][j] = depthTemp;
               }
            }
            //printf("%d %d %d %d\n", boundingBoxes[k].xl, boundingBoxes[k].yl, boundingBoxes[k].xr, boundingBoxes[k].yr );
         }
      }
   }
   return 0;
}
int rasterize2( BasicModel &mesh, Tga &file )
{
   Tga::pixel **data = file.getBuffer();
   int width = file.getWidth();
   int height = file.getHeight();
   float depth[height][width];
   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         depth[i][j] = 100000;
   int tris = mesh.Triangles.size();
   Vertex *vertices = convertVertices( mesh, width, height );
   printf("w: %d %d\n", vertices[1].x, vertices[2].y );
   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );

   for( int k = 0; k < tris; ++k )
   {
      for( int j = boundingBoxes[k].xl; j <= boundingBoxes[k].xr; ++j )
      {
         for( int i = boundingBoxes[k].yl; i <= boundingBoxes[k].yr; ++i )
         {
            float xa = vertices[triangles[k].a].x;
            float ya = vertices[triangles[k].a].y;
            float xc = vertices[triangles[k].c].x;
            float yc = vertices[triangles[k].c].y;
            float yb = vertices[triangles[k].b].y;
            float xb = vertices[triangles[k].b].x;
            float beta = ((xa-xc)*(i-yc) - (j-xc)*(ya-yc))/((xc-xa)*(yb-ya) - (xb-xa)*(yc-ya));
            float gamma = ((xb-xa)*(i-ya) - (j-xa)*(yb-ya))/((xb-xa)*(yc-ya) - (xc-xa)*(yb-ya));
            float alpha;
            if( beta <= 1 && gamma <= 1 && beta+gamma <= 1 ){
               alpha = 1- beta -gamma;
            }
            else
               continue;

            float depthTemp = alpha*vertices[triangles[k].a].z + beta*vertices[triangles[k].b].z + gamma*vertices[triangles[k].c].z;
            if( depthTemp < depth[i][j] )
            {
               data[i][j].r = alpha*triangles[k].color.r + beta*triangles[k].color.r
                  + gamma*triangles[k].color.r;
               data[i][j].g = alpha*triangles[k].color.g + beta*triangles[k].color.g
                  + gamma*triangles[k].color.g;
               data[i][j].b = alpha*triangles[k].color.b + beta*triangles[k].color.b
                  + gamma*triangles[k].color.b;
               depth[i][j] = depthTemp;
            }
         }
      }
   }
   return 0;
}
Triangle *createTriangles( BasicModel &mesh, BoundingBox **boundingBoxes, Vertex *screenVerts )
{
   Triangle *triangles = (Triangle *) malloc( sizeof(Triangle) * mesh.Triangles.size() );
   *boundingBoxes = (BoundingBox *) malloc( sizeof(BoundingBox) * mesh.Triangles.size() );
   std::vector<Model::Tri *>::iterator it;
   int i = 0;

   for( it = mesh.Triangles.begin(); it < mesh.Triangles.end(); it++ )
   {
      triangles[i].a = (**it).v1-1;
      triangles[i].b = (**it).v2-1;
      triangles[i].c = (**it).v3-1;
      createBB( (*boundingBoxes)[i], triangles[i], screenVerts );
      i++;
   }
   return triangles;
}
void createBB( BoundingBox &box, const Triangle &triangle, Vertex *verts )
{
   //lower left
   box.xl = verts[triangle.a].x;
   box.yl = verts[triangle.a].y;
   //upper right
   box.xr = verts[triangle.a].x;
   box.yr = verts[triangle.a].y;

   if( box.xl > verts[triangle.b].x )
      box.xl = verts[triangle.b].x;
   if( box.yl > verts[triangle.b].y )
      box.yl = verts[triangle.b].y;
   if( box.xl > verts[triangle.c].x )
      box.xl = verts[triangle.c].x;
   if( box.yl > verts[triangle.c].y )
      box.yl = verts[triangle.c].y;

   if( box.xr < verts[triangle.b].x )
      box.xr = verts[triangle.b].x;
   if( box.yr < verts[triangle.b].y )
      box.yr = verts[triangle.b].y;
   if( box.xr < verts[triangle.c].x )
      box.xr = verts[triangle.c].x;
   if( box.yr < verts[triangle.c].y )
      box.yr = verts[triangle.c].y;

   printf("%d, %d, %d \n", triangle.a, triangle.b, triangle.c);
   printf("%d, %d, %d \n", verts[triangle.a].x, verts[triangle.b].x, verts[triangle.c].x);
   printf("%d, %d, %d , %d\n", box.xl, box.yl, box.xr, box.yr);
}
Vertex *convertVertices( BasicModel &mesh, int width, int height )
{
   Vertex *verts = (Vertex *) malloc(sizeof(Vertex) * mesh.Vertices.size());
   std::vector<Vector3 *>::iterator it;
   int i = 0;

   for( it = mesh.Vertices.begin(); it < mesh.Vertices.end(); it++ )
   {
      verts[i].x = ((width-1)/2) * ((*it)->x) + (width-1)/2;
      verts[i].y = ((height-1)/2) * ((*it)->y) + (height-1)/2;
      verts[i].z = (*it)->z*width*height;
      printf("putting %d: %d, %d, %d \n", i, verts[i].x, verts[i].y, verts[i].z);
      i++;
   }
   return verts;
}

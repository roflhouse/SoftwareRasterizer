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
   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         depth[i][j] = 100000;
   int tris = mesh.Triangles.size();

   //Converts mesh verts to screenspace
   Vertex *vertices = convertVertices( mesh, width, height );

   //printf("w: %d %d\n", vertices[1].x, vertices[2].y );
   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );

   for( int k = 0; k < tris; ++k )
   {
      for( int j = boundingBoxes[k].xl; j <= boundingBoxes[k].xr; ++j )
      {
         if( j >= width)
            break;
         if( j < 0 )
         {
            j=-1;
            continue;
         }
         for( int i = boundingBoxes[k].yl; i <= boundingBoxes[k].yr; ++i )
         {
            if( i >= height)
               break;
            if( i < 0 )
            {
               i = -1;
               continue;
            }
            float xa = vertices[triangles[k].a].x;
            float ya = vertices[triangles[k].a].y;
            float xb = vertices[triangles[k].b].x;
            float yb = vertices[triangles[k].b].y;
            float xc = vertices[triangles[k].c].x;
            float yc = vertices[triangles[k].c].y;
            float beta = ((xa-xc)*(i-yc) - (j-xc)*(ya-yc))/((xb-xa)*(yc-ya) - (xc-xa)*(yb-ya));
            float gamma = ((xb-xa)*(i-ya) - (j-xa)*(yb-ya))/((xb-xa)*(yc-ya) - (xc-xa)*(yb-ya));
            float alpha;
            //printf("i: %d j: %d, a: %f, %f %f %f %f %f\n", i, j, xa, ya, xb, yb, xc, yc );
            //printf("i: %d j: %d, a: %f, %f %f  %f %f\n", i, j, alpha, beta, gamma, ((xa-xc)*(i-yc) - (j-xc)*(ya-yc)), ((xc-xa)*(yb-ya) - (xb-xa)*(yc-ya)) );
            if( beta+gamma <= 1 && beta >=0 && gamma >= 0 ){
               alpha = 1- beta -gamma;
            }
            else
               continue;

            float depthTemp = alpha*vertices[triangles[k].a].z + beta*vertices[triangles[k].b].z + gamma*vertices[triangles[k].c].z;
            if( depthTemp < depth[i][j] )
            {
               /*data[i][j].r = alpha*triangles[k].color.r + beta*triangles[k].color.r
                 + gamma*triangles[k].color.r;
                 data[i][j].g = alpha*triangles[k].color.g + beta*triangles[k].color.g
                 + gamma*triangles[k].color.g;
                 data[i][j].b = alpha*triangles[k].color.b + beta*triangles[k].color.b
                 + gamma*triangles[k].color.b;
                 */
               data[i][j].r = alpha * 1 + beta * 0 + gamma * 0;
               data[i][j].g = alpha * 0 + beta * 1 + gamma * 0;
               data[i][j].b = alpha * 0 + beta * 0 + gamma * 1;
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

   //printf("%d, %d, %d \n", triangle.a, triangle.b, triangle.c);
   //printf("%d, %d, %d \n", verts[triangle.a].x, verts[triangle.b].x, verts[triangle.c].x);
   //printf("BB:%d, %d, %d , %d\n", box.xl, box.yl, box.xr, box.yr);
}
Vertex *convertVertices( BasicModel &mesh, int width, int height )
{
   Vertex *verts = (Vertex *) malloc(sizeof(Vertex) * mesh.Vertices.size());
   std::vector<Vector3 *>::iterator it;
   int i = 0;
   glm::mat4 transform = glm::scale( glm::mat4(1.0f), glm::vec3( 7,7, 0) );
   glm::vec4 cent = glm::vec4( (float)-mesh.center.x, (float)-mesh.center.y, 0.0, 1.0 );
   transform = glm::translate( transform, glm::vec3( cent[0], cent[1], 0 ) ); 
   //glm::mat4 transform = glm::mat4(1.0f);

   for( it = mesh.Vertices.begin(); it < mesh.Vertices.end(); it++ )
   {
      glm::vec4 p = glm::vec4(((*it)->x), ((*it)->y), ((*it)->z), 1.0f);
      p = transform * p;

      verts[i].x = ((width-1)/2) * p[0] + (width-1)/2;
      verts[i].y = ((height-1)/2) * p[1] + (height-1)/2;
      verts[i].z = p[2]*width*height;
      //printf("putting %d: %d, %d, %d \n", i, verts[i].x, verts[i].y, verts[i].z);
      i++;
   }
   return verts;
}

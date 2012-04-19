/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "RasterizeFuncs.h"
#include "../NewMeshParser/utils.h"
#include <vector>

int rasterize( BasicModel &mesh, Tga &file )
{
   Normal light;
   light.x = 3;
   light.y = 3;
   light.z = 3;
   Tga::pixel **data = file.getBuffer();
   int width = file.getWidth();
   int height = file.getHeight();
   float depth[height][width];
   for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
         depth[i][j] = -100000;
   int tris = mesh.Triangles.size();

   //Converts mesh verts to screenspace
   Vertex *vertices = convertVertices( mesh, width, height );

   //printf("w: %d %d\n", vertices[1].x, vertices[2].y );
   BoundingBox *boundingBoxes;
   Triangle *triangles = createTriangles( mesh, &boundingBoxes, vertices );
   Normal *normals = createNormals( mesh, (int)mesh.Vertices.size() );
   printf("Number: %d\n", mesh.Vertices.size());

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
            if( beta+gamma <= 1 && beta >=0 && gamma >= 0 ){
               alpha = 1- beta -gamma;
            }
            else
               continue;

            float depthTemp = mesh.Vertices[mesh.Triangles[k]->v1-1]->z *alpha+
               mesh.Vertices[mesh.Triangles[k]->v2-1]->z *beta + mesh.Vertices[mesh.Triangles[k]->v3-1]->z *gamma;
            //alpha*vertices[triangles[k].a].z + beta*vertices[triangles[k].b].z + gamma*vertices[triangles[k].c].z;
            if( depthTemp > depth[i][j] )
            {
               Color colorA = calcLighting( normals, mesh, triangles[k].a, light );
               Color colorB = calcLighting( normals, mesh, triangles[k].b, light );
               Color colorC = calcLighting( normals, mesh, triangles[k].c, light );

               //data[i][j].r = alpha * colorA.r + beta * colorB.r + gamma * colorC.r;
               //data[i][j].g = alpha * colorA.g + beta * colorB.g + gamma * colorC.g;
               //data[i][j].b = alpha * colorA.b + beta * colorB.b + gamma * colorC.b;
               data[i][j].r = colorA.r*alpha + colorB.r*beta + colorC.r*gamma;
               data[i][j].g = colorA.g*alpha + colorB.g*beta + colorC.g*gamma;
               data[i][j].b = colorA.b*alpha + colorB.b*beta+ colorC.b*gamma;
               /*Normal li;
               li.x = light.x - mesh.Vertices[mesh.Triangles[k]->v1-1]->x;
               li.y = light.y - mesh.Vertices[mesh.Triangles[k]->v2-1]->y;
               li.z = light.z - mesh.Vertices[mesh.Triangles[k]->v3-1]->z;
               Normal n;
               n.x = mesh.Triangles[k]->normal.x;
               n.y = mesh.Triangles[k]->normal.y;
               n.z = mesh.Triangles[k]->normal.z;
               li = normalize(li);
               float hh = dot( n, li );
               if( hh < 0 )
                  hh = 0;
               if( hh > 1 )
                  hh = 1;
               data[i][j].r = hh*.8;
               data[i][j].g = hh*.8;
               data[i][j].b = hh*.8;
               //printf("%f %f %f color\n", mesh.Vertices[triangles[0].a]->x, mesh.Vertices[triangles[0].a]->y, mesh.Vertices[triangles[0].a]->z);
               //printf("%f %f %f color\n", data[i][j].r, data[i][j].g, data[i][j].b);
*/
               depth[i][j] = depthTemp;
            }
         }
      }
   }
   return 0;
}
Color calcLighting( Normal *ns, BasicModel &mesh, int v, Normal l )
{
   Normal li;
   Color ret;
   li.x = l.x - mesh.Vertices[v]->x;
   li.y = l.y - mesh.Vertices[v]->y;
   li.z = l.z - mesh.Vertices[v]->z;
   li = normalize( li );
   Normal n = normalize( ns[v] );
   float dotpro = dot( n, li );
   if( dotpro < 0 )
   {
      ret.r = 0;
      ret.g = 0;
      ret.b = 0;
   }
   else if(dotpro > 1){
      ret.r = .8;
      ret.g = .8;
      ret.b = .8;
   }
   else{
      ret.r =  dotpro * .8;
      ret.g =  dotpro * .8;
      ret.b =  dotpro * .8;
   }
   //if( ret.r + ret.g +ret.b <= 0.001 )
   ///   printf("NORMAL: %f %f %f %f\n", ns[v].x, ns[v].y, ns[v].z, dotpro);
   return ret;
}
Normal normalize( Normal n )
{
   float total = n.x*n.x + n.y*n.y + n.z*n.z;
   total = sqrt(total);
   n.x = n.x/total;
   n.y = n.y/total;
   n.z = n.z/total;
   return n;
}
float dot( Normal n1, Normal n2 )
{
   return n1.x*n2.x + n1.y*n2.y + n1.z*n2.z;
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
      verts[i].z = p[2]*10000;
      //printf("putting %d: %d, %d, %d \n", i, verts[i].x, verts[i].y, verts[i].z);
      i++;
   }
   return verts;
}
Normal *createNormals( BasicModel &mesh, int num_verts )
{
   Normal *normals = (Normal *) malloc( sizeof(Normal) * num_verts );
   //int numbers[num_verts];
   for( int i = 0; i< num_verts; i++)
   {
      normals[i].x = 0;
      normals[i].y = 0;
      normals[i].z = 0;
   }

   for( unsigned int i = 0; i < mesh.Triangles.size(); i++ )
   {
      Vector3 tn = mesh.Triangles[i]->normal;
      int a = mesh.Triangles[i]->v1-1;
      int b = mesh.Triangles[i]->v2-1;
      int c = mesh.Triangles[i]->v3-1;
      normals[a].x += tn.x;
      normals[a].y += tn.y;
      normals[a].z += tn.z;
      normals[b].x += tn.x;
      normals[b].y += tn.y;
      normals[b].z += tn.z;
      normals[c].x += tn.x;
      normals[c].y += tn.y;
      normals[c].z += tn.z;
      normals[a].x /= 2;
      normals[a].y /= 2;
      normals[a].z /= 2;
      normals[b].x /= 2;
      normals[b].y /= 2;
      normals[b].z /= 2;
      normals[c].x /= 2;
      normals[c].y /= 2;
      normals[c].z /= 2;
   }
   return normals;
}

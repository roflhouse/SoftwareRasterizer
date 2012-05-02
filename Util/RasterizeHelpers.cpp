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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Color *createColors( BasicModel &mesh, Normal *normals, Normal light )
{
   Color *colors = (Color *) malloc( sizeof(Color) * mesh.Vertices.size() );
   for( unsigned int i = 0; i < mesh.Vertices.size(); i++ )
   {
      colors[i] = calcLighting( normals, mesh, i, light );
   }
   return colors;
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
   glm::mat4 transform = glm::scale( glm::mat4(1.0f), glm::vec3( 10, 10, 0) );
   glm::vec4 cent = glm::vec4( (float)-mesh.center.x, (float)-mesh.center.y, 0.0, 1.0 );
   transform = glm::translate( transform, glm::vec3( cent[0], cent[1], 0 ) );
//   transform = glm::translate( transform, glm::vec3( -0.3, -0.3, 0 ) );
   //glm::mat4 transform = glm::mat4(1.0f);

   for( it = mesh.Vertices.begin(); it < mesh.Vertices.end(); it++ )
   {
      glm::vec4 p = glm::vec4(((*it)->x), ((*it)->y), ((*it)->z), 1.0f);
      p = transform * p;

      verts[i].x = ((width-1)/2) * p[0] + (width-1)/2;
      verts[i].y = ((height-1)/2) * p[1] + (height-1)/2;
      verts[i].z = ((*it)->z);
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

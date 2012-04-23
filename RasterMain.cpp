/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "RasterMain.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

int width_of_image;
int height_of_image;

int main(int argc, char *argv[])
{
   glm::mat4 transforms = glm::mat4(1.0f);
   transforms = glm::translate(transforms, glm::vec3( 20, 0 , 0) );

   BasicModel mesh( parseCommandLine(argc, argv) );

   Tga file( width_of_image, height_of_image );

   pixel p;
   p.r = 0;
   p.g = 1;
   p.b = 0;
   pixel **data = file.getBuffer();

   for( int i = 0; i < height_of_image; i++ ){
      for( int j = 0; j < width_of_image; j++ )
         data[i][j] = p;
   }

   rasterize( mesh, file );

   file.writeTga( "output.tga" );

   printf("Image w%d h%d , %d %d %d\n", width_of_image, height_of_image, (int)mesh.Triangles.size(),
         (int)mesh.Vertices.size(), (int)mesh.VerticesNormals.size() ); 

   return EXIT_SUCCESS;
}
char *parseCommandLine(int argc, char *argv[])
{
   if (argc >= 3 )
   {
      if( argv[1][0] == '+' && (argv[1][1] == 'W' || argv[1][1] == 'w') )
      {
         char *temp = &(argv[1][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> width_of_image;
         if (!s )
         {
            printf("Input Error width unknown\n");
            exit(1);
         }
      }
      if( argv[2][0] == '+' && (argv[2][1] == 'H' || argv[2][1] == 'h') )
      {
         char *temp = &(argv[2][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> height_of_image;
         if (!s)
         {
            printf("Input Error height unknown\n");
            exit(1);
         }
      }
      if (width_of_image <= 0 || height_of_image <= 0)
      {
         printf("Input Error invalid demenstions, width: %d, height: %d\n", width_of_image, height_of_image);
         exit(1);
      }
      if( argc > 3 )
      {
         return argv[3];
      }
   }
   printf("Error miss use of raytracer: raytracer +w#### +h#### filename.pov\n");
   exit(EXIT_FAILURE);
}


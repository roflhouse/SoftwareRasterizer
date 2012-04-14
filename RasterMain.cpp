/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#include "RasterMain.h"

int main(int argc, char *argv[])
{
   Tga::Tga file( 200, 200 );
   Tga::pixel p;
   p.r = 1;
   p.g = 0;
   p.b = 0;
   
   for( int i = 0; i < 200; i++ ){
     for( int j = 0; j < 200; j++ )
       file.setPixel( i, j, p );
   }
   file.writeTga( "test.tga" );
   
   return EXIT_SUCCESS;
}

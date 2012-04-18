/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef TGA_H
#define TGA_H
#include <stdlib.h>
#include <string>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>

#include "Header.h"
class Tga
{
    public:
        typedef struct {
           float r;
           float g;
           float b;
        } pixel;

        Tga( short int w, short int h );
        ~Tga();
        int writeTga(std::string filename);
        void setPixel(int width, int height, pixel p);
        void setPixels( int width, int height, pixel **p);
        pixel **getBuffer( );
        int getWidth();
        int getHeight();
    private:
        pixel **data;
        Header *header;
        short int width;
        short int height;
};
#endif

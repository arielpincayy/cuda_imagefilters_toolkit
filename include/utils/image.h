#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include "include/stb/stb_image.h"
#include "include/stb/stb_image_write.h"

typedef struct {
    stbi_uc *data;
    int width, height, channels;
} Image;

#ifdef __cplusplus
extern "C"{
#endif

Image *image2arr(const char *file);
void saveImg(Image *img, const char *filename);
Image *createImg(int width, int height, int channels);  
void freeImage(Image *img);

#ifdef __cplusplus
}
#endif

#endif

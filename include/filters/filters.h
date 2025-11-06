#ifndef FILTERS_H
#define FILTERS_H

#include "include/utils/image.h"
#include <cuda_runtime.h>

// Funciones de procesamiento
Image* rgb2gray(Image* img);
Image* GaussianBlur(Image* img, int kernel_dim, float stdev);
Image* Sobel(Image* img);

#endif // FILTERS_H

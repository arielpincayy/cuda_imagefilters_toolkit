#ifndef FILTERS_H
#define FILTERS_H

#include "include/utils/image.h"
#include <cuda_runtime.h>

// --- INICIO DEL CAMBIO ---
#ifdef __cplusplus
extern "C" {
#endif

// Funciones de procesamiento
Image* rgb2gray(Image* img);
Image* GaussianBlur(Image* img, int kernel_dim, float stdev);
Image* Sobel(Image* img);

#ifdef __cplusplus
}
#endif
// --- FIN DEL CAMBIO ---

#endif // FILTERS_H
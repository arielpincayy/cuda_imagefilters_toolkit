#include "include/filters/filters.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define THREADS 16
#define MAX_KERNEL_DIM 15
#define MAX_KERNEL_SIZE MAX_KERNEL_DIM*MAX_KERNEL_DIM

__constant__ float d_kernel[MAX_KERNEL_SIZE];
__constant__ float d_GX[9];
__constant__ float d_GY[9];


__global__ void rgb2gray_kernel(stbi_uc *gray_image, stbi_uc *image, int width, int height, int channels){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    stbi_uc r,g,b, gray;
    unsigned int rgbOffset;

    if(idx < width * height){
        rgbOffset = idx * channels;

        r = image[rgbOffset];
        g = image[rgbOffset + 1];
        b = image[rgbOffset + 2];

        gray = (stbi_uc)(0.299f * r + 0.587f * g + 0.114f * b);

        gray_image[idx] = gray;
    }

}

float *gaussianKernel(int n, float stdev){
    float *kernel = (float *)malloc(n*n*sizeof(float));
    int i, j;
    float vars = stdev*stdev;
    float frac = 1/(sqrt(2*M_PI*vars));

    int mid = n/2;
    
    float sum = 0;
    for(i=-mid; i<=mid; i++){
        for(j=-mid; j<=mid; j++){
            kernel[(j+mid)*n + (i+mid)] = frac * expf(-(((i*i) + (j*j)) / (2.0f * vars)));
            sum += kernel[(j+mid)*n + (i+mid)];
        }
    }

    for(i=0; i<n*n; i++) kernel[i] /= sum;

    return kernel;
}
__device__ void Convolution(stbi_uc *convol_image, const stbi_uc *image, int width, int height, int channels, int kdim, 
                            int tx, int ty, int bx, int by, int blockW, int blockH, float *kernel, float *sTile) {

    int x = bx * blockW + tx;
    int y = by * blockH + ty;

    if (x >= width || y >= height) return;

    int mid = kdim / 2;
    
    int sPitch = blockW + 2 * mid;

    #define ST_INDEX(srow, scol, ch) (((srow) * sPitch + (scol)) * 4 + (ch))
    
    int tileImageStartX = bx * blockW - mid;
    int tileImageStartY = by * blockH - mid;

    int tileCols = blockW + 2 * mid;
    int tileRows = blockH + 2 * mid;

    int tx_flat = ty * blockW + tx;
    int threadsPerBlock = blockW * blockH;
    int totalTileCells = tileRows * tileCols;

    int imgX, imgY, clampedX, clampedY, local_row, local_col, c, offset_image;
    for (int idx = tx_flat; idx < totalTileCells; idx += threadsPerBlock) {
        local_row = idx / tileCols;
        local_col = idx % tileCols;

        imgX = tileImageStartX + local_col;
        imgY = tileImageStartY + local_row;

        clampedX = MIN(MAX(imgX, 0), width - 1);
        clampedY = MIN(MAX(imgY, 0), height - 1);

        offset_image = clampedY * width + clampedX;
        for (c = 0; c < 4; ++c) {
            if (channels > c) {
                sTile[ ST_INDEX(local_row, local_col, c) ] = (float) image[(offset_image) * channels + c];
            } else {
                sTile[ ST_INDEX(local_row, local_col, c) ] = 0.0f;
            }
        }
    }

    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int kx, ky, base;
    float kval;
    for (ky = 0; ky < kdim; ky++) {
        for (kx = 0; kx < kdim; kx++) {
            kval = kernel[ ky * kdim + kx ];
            base = ((ty + ky) * sPitch + (tx + kx)) * 4;
            for(c=0; c<channels; c++){
                acc[c] += sTile[base + c] * kval;
            }
        }
    }

    int outIdx = (y * width + x) * channels;
    float v;
    for (c = 0; c < channels; c++) {
        v = MIN(MAX(acc[c], 0), 255);
        convol_image[outIdx + c] = (stbi_uc) (v + 0.5f);
    }

    #undef ST_INDEX
}

__global__ void GaussianBlur_kernel(stbi_uc* output, const stbi_uc* input, int width, int height, int channels, int kdim) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int blockW = blockDim.x;
    int blockH = blockDim.y;

    int x = bx * blockW + tx;
    int y = by * blockH + ty;

    if (x < width && y < height) {
        extern __shared__ float sTile[];
        Convolution(output, input, width, height, channels, kdim, tx, ty, bx, by, blockW, blockH, d_kernel, sTile);
    }
}

__global__ void Sobel_kernel(stbi_uc* output, const stbi_uc* input, stbi_uc *GX_image, stbi_uc *GY_image, int width, int height){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int blockW = blockDim.x;
    int blockH = blockDim.y;

    int x = bx * blockW + tx;
    int y = by * blockH + ty;

    if (x >= width || y >= height) return;

    extern __shared__ float sTile[];

    Convolution(GX_image, input, width, height, 1, 3, tx, ty, bx, by, blockW, blockH, d_GX, sTile);
    Convolution(GY_image, input, width, height, 1, 3, tx, ty, bx, by, blockW, blockH, d_GY, sTile);

    __syncthreads();

    int pos = y * width + x;
    output[pos] = sqrtf(GX_image[pos] * GX_image[pos] + GY_image[pos] * GY_image[pos]);

}

Image *rgb2gray(Image *img) {
    Image *gray_image = createImg(img->width, img->height, 1);
    stbi_uc *image_h = img->data;
    stbi_uc *gray_image_d, *image_d;

    cudaMalloc((void **)&gray_image_d, gray_image->height * gray_image->width * sizeof(stbi_uc));
    cudaMalloc((void **)&image_d, img->height * img->width * img->channels * sizeof(stbi_uc));
    cudaMemcpy(image_d, image_h, img->height * img->width * img->channels * sizeof(stbi_uc), cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADS * THREADS;
    int blocks = (img->height * img->width + threadsPerBlock - 1) / threadsPerBlock;

    // --- Medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rgb2gray_kernel<<<blocks, threadsPerBlock>>>(gray_image_d, image_d, img->width, img->height, img->channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[Kernel rgb2gray] Tiempo de ejecución: %f ms\n", milliseconds);

    cudaMemcpy(gray_image->data, gray_image_d, gray_image->height * gray_image->width * sizeof(stbi_uc), cudaMemcpyDeviceToHost);
    cudaFree(gray_image_d);
    cudaFree(image_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gray_image;
}

Image *Sobel(Image *gray_image) {

    if(gray_image->channels > 1 ){
        perror("Image must be in gray scale\n");
        exit(1);
    }

    const float GX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float GY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    Image *sobel_image = createImg(gray_image->width, gray_image->height, 1);

    size_t size_image = gray_image->height * gray_image->width * sizeof(stbi_uc);

    stbi_uc *d_sobelImage, *d_grayImage, *d_GXImage, *d_GYImage;

    cudaMalloc(&d_sobelImage, size_image);
    cudaMalloc(&d_grayImage, size_image);
    cudaMalloc(&d_GXImage, size_image);
    cudaMalloc(&d_GYImage, size_image);
    cudaMemcpy(d_grayImage, gray_image->data, size_image, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_GX, GX, 9*sizeof(float));
    cudaMemcpyToSymbol(d_GY, GY, 9*sizeof(float));
    
    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 blocks((gray_image->width + threadsPerBlock.x - 1) / threadsPerBlock.x, (gray_image->height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int sharedMemSize = (THREADS + 6) * (THREADS + 6) * 4 * sizeof(float);

    // --- Medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    Sobel_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_sobelImage, d_grayImage, d_GXImage, d_GYImage, gray_image->width, gray_image->height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[Kernel Sobel] Tiempo de ejecución: %f ms\n", milliseconds);

    cudaMemcpy(sobel_image->data, d_sobelImage, size_image, cudaMemcpyDeviceToHost);
    cudaFree(d_grayImage);
    cudaFree(d_GXImage);
    cudaFree(d_GYImage);
    cudaFree(d_sobelImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return sobel_image;
}

Image *GaussianBlur(Image *img, int kernel_dim, float stdev) {
    Image *blurerImg = createImg(img->width, img->height, img->channels);
    stbi_uc *d_data, *d_blurerData;
    float *h_kernel;

    cudaMalloc(&d_data, sizeof(stbi_uc) * img->width * img->height * img->channels);
    cudaMalloc(&d_blurerData, sizeof(stbi_uc) * blurerImg->width * blurerImg->height * blurerImg->channels);
    cudaMemcpy(d_data, img->data, sizeof(stbi_uc) * img->width * img->height * img->channels, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS, THREADS);
    dim3 blocks((img->width + threadsPerBlock.x - 1) / threadsPerBlock.x, (img->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    h_kernel = gaussianKernel(kernel_dim, stdev);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_dim * kernel_dim * sizeof(float));
    int sharedMemSize = (THREADS + 2 * kernel_dim) * (THREADS + 2 * kernel_dim) * 4 * sizeof(float);

    // --- Medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GaussianBlur_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_blurerData, d_data, img->width, img->height, img->channels, kernel_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[Kernel GaussianBlur] Tiempo de ejecución: %f ms\n", milliseconds);

    cudaMemcpy(blurerImg->data, d_blurerData, sizeof(stbi_uc) * blurerImg->width * blurerImg->height * blurerImg->channels, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_blurerData);
    free(h_kernel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return blurerImg;
}
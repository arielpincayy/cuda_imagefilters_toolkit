#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <include/utils/image.h>

Image *image2arr(const char *file) {
    int width, height, channels;

    stbi_uc *img = stbi_load(file, &width, &height, &channels, 0);
    if (img == NULL) {
        perror("Error loading image\n");
        exit(1);
    }

    Image *image_arr = (Image *)malloc(sizeof(Image));
    if (image_arr == NULL) {
        perror("Error allocating memory for image array\n");
        exit(1);
    }

    image_arr->width = width;
    image_arr->height = height;
    image_arr->channels = channels;
    image_arr->data = img;

    return image_arr;
}

void saveImg(Image *img, const char *filename) {
    int width = img->width, height = img->height, channels = img->channels;

    stbi_write_jpg(filename, width, height, channels, img->data, 100);
}

Image *createImg(int width, int height, int channels){
    Image *img = (Image *)malloc(sizeof(Image));

    if (img == NULL) {
        perror("Error allocating memory for image array\n");
        exit(1);
    }

    img->height = height;
    img->width = width;
    img->channels = channels;
    img->data = (stbi_uc *)malloc(width*height*channels*sizeof(stbi_uc));
    return img;
}

void freeImage(Image *img) {
    free(img->data);
}
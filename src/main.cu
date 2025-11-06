#include <include/filters/filters.h>
#include <string.h>


int main(int argc, char **argv){

    if (argc < 4) {
        printf("Usage: %s <image_path> <output_path_gray> <output_path_sobel> <output_path_gaussian>\n", argv[0]);
        return -1;
    }

    char *filename_in = argv[1];
    char *filename_out_gray = argv[2];
    char *filename_out_sobel = argv[3];
    char *filename_out_gaussian = argv[4];

    Image *img = image2arr(filename_in);
    Image *gray = rgb2gray(img);
    Image *sobel = Sobel(gray);
    Image *gaussian = GaussianBlur(img, 9, 10.0f);

    saveImg(gaussian, filename_out_gaussian);
    saveImg(gray, filename_out_gray);
    saveImg(sobel, filename_out_sobel);
    
    freeImage(gray);
    freeImage(gaussian);
    freeImage(sobel);
    freeImage(img);

    return 0;
}
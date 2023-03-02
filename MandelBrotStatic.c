#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITERATIONS 1000

typedef struct {
    double real;
    double imag;
} complex;

int mandelbrot(complex c) {
    int i;
    complex z = {0.0, 0.0};
    for (i = 0; i < MAX_ITERATIONS; i++) {
        if (z.real * z.real + z.imag * z.imag > 4.0) {
            return i;
        }
        complex temp;
        temp.real = z.real * z.real - z.imag * z.imag + c.real;
        temp.imag = 2 * z.real * z.imag + c.imag;
        z = temp;
    }
    return MAX_ITERATIONS;
}

int main() {
    int i, j, color;
    complex c;
    double x_min = -2.0, x_max = 2.0;
    double y_min = -1.5, y_max = 1.5;
    double x, y;
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;
    int *image = (int*) malloc(WIDTH * HEIGHT * sizeof(int));
    #pragma omp parallel shared(image) private(i, j, x, y, c, color)
    #pragma omp for schedule(static)
    for (i = 0; i < HEIGHT; i++) {
        y = y_min + i * dy;
        for (j = 0; j < WIDTH; j++) {
            x = x_min + j * dx;
            c.real = x;
            c.imag = y;
            color = mandelbrot(c);
            image[i * WIDTH + j] = color;
        }
    }
    FILE *fp = fopen("mandelbrot.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            color = image[i * WIDTH + j];
            fputc(color % 256, fp);
            fputc((color / 256) % 256, fp);
            fputc((color / 65536) % 256, fp);
        }
    }
    fclose(fp);
    free(image);
    return 0;
}

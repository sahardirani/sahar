#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int main(int argc, char** argv) {

    int rows, cols;
    double xmin, xmax, ymin, ymax;
    double x, y, x0, y0;
    int max_iter, iter, i, j;
    if (argc != 8) {
        printf("Usage: %s rows cols xmin xmax ymin ymax max_iter\n", argv[0]);
        return 0;
    }
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    xmin = atof(argv[3]);
    xmax = atof(argv[4]);
    ymin = atof(argv[5]);
    ymax = atof(argv[6]);
    max_iter = atoi(argv[7]);

    for (i = 0; i < rows; i++) {
        y = ymin + i * (ymax - ymin) / rows;
        for (j = 0; j < cols; j++) {
            x = xmin + j * (xmax - xmin) / cols;
            x0 = x;
            y0 = y;
            iter = 0;
            while (x * x + y * y <= 4 && iter < max_iter) {
                double complex z = x + y * I;
                double complex z0 = x0 + y0 * I;
                x = creal(z * z) + creal(z0);
                y = cimag(z * z) + cimag(z0);
                iter++;
            }
            printf("%d ", iter);
        }
        printf("\n");
    }

    return 0;
}

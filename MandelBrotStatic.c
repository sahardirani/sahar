#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#define MASTER 0

int main(int argc, char** argv) {
    int rank, size;
    int rows, cols;
    int chunk_size, remainder;
    double xmin, xmax, ymin, ymax;
    double x, y, x0, y0;
    int max_iter, iter, i, j, k;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 8) {
        if (rank == MASTER) {
            printf("Usage: %s rows cols xmin xmax ymin ymax max_iter\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    xmin = atof(argv[3]);
    xmax = atof(argv[4]);
    ymin = atof(argv[5]);
    ymax = atof(argv[6]);
    max_iter = atoi(argv[7]);
    chunk_size = rows / size;
    remainder = rows % size;
    int start_row = rank * chunk_size;
    int end_row = start_row + chunk_size - 1;
    if (rank == size - 1) {
        end_row += remainder;
    }
    int chunk_rows = end_row - start_row + 1;
    int chunk = (int)malloc(chunk_rows * cols * sizeof(int));

   
    for (i = start_row; i <= end_row; i++) {
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
            chunk[(i - start_row) * cols + j] = iter;
        }
    }
    if (rank == MASTER) {
        int image = (int)malloc(rows * cols * sizeof(int));
        MPI_Gather(chunk, chunk_rows * cols, MPI_INT, image, chunk_rows * cols, MPI_INT, MASTER, MPI_COMM_WORLD);
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                printf("%d ", image[i * cols + j]);
            }
            printf("\n");
        }
        free(image);
    } else {
        MPI_Gather(chunk, chunk_rows * cols, MPI_INT, NULL, chunk_rows * cols, MPI_INT, MASTER, MPI_COMM_WORLD);
    }
    free(chunk);
    MPI_Finalize();
    return 0;
}

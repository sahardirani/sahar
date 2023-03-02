#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <float.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include "timer.h"
#include "cmdline.h"
#define N   2           /* size of problem space x, y from -N to N */
#define NPIXELS     800         /* display window size in pixels */
#define WORK_TAG    1           /* master to worker message */
#define DATA_TAG    2           /* worker to master message */
#define STOP_TAG    3           /* master to worker message */
#include "mandelbrot-gui.h"    /* has setup(), interact() */

typedef struct {
    double real;
    double imag;
} complex;

int master_pgm(int nworkers, int width, int height, double real_min, double real_max, double imag_min, double imag_max, int maxiter);
int worker_pgm(int myID, int width, int height, double real_min, double real_max, double  imag_min, double imag_max, int maxiter);
int main (int argc, char *argv[]) {
    int nprocs;
    int myid;
    int returnval;
    int maxiter;
    double real_min = -N;
    double real_max = N;
    double imag_min = -N;
    double imag_max = N;
    int width = NPIXELS;
    int height = NPIXELS;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization error\n");
        exit(EXIT_FAILURE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (nprocs < 2) {
        fprintf(stderr, "Number of processes must be at least 2\n");
        MPI_Finalize(); exit(EXIT_FAILURE);
    }
    char *usage_msg = "usage:  %s maxiter [x0 y0 size]\n";
    maxiter = get_integer_arg_extended(argc, argv, 1, 1, "maxiter", usage_msg, (myid == 0), (void (*)(void))MPI_Finalize);
    if (argc > 2) {
        double x0 = get_floating_arg_extended(argc, argv, 2, -DBL_MAX, "x0", usage_msg,            (myid == 0), (void (*)(void))MPI_Finalize);
        double y0 = get_floating_arg_extended(argc, argv, 3, -DBL_MAX, "y0", usage_msg,
                (myid == 0), (void (*)(void))MPI_Finalize);
        double size = get_floating_arg_extended(argc, argv, 4, 0, "size", usage_msg,                (myid == 0), (void (*)(void))MPI_Finalize);
        real_min = x0 - size;
        real_max = x0 + size;
        imag_min = y0 - size;
        imag_max = y0 + size;
    }
if (myid == 0) {
        returnval = master_pgm(nprocs-1, width, height,
			real_min, real_max, imag_min, imag_max, maxiter);
    }
    else {
        returnval = worker_pgm(myid, width, height,
			real_min, real_max, imag_min, imag_max, maxiter);
    }
    MPI_Finalize();
    return returnval;
}
int master_pgm(int nworkers, int width, int height,
        double real_min, double real_max,
        double imag_min, double imag_max,
        int maxiter) {

    Display *display;
    Window win;
    GC gc;
    long min_color = 0, max_color = 0;
    int this_row, next_row;
    double start_time, end_time;
    long *data_msg = malloc((width+1) * sizeof(*data_msg));
    MPI_Status status;
    int tasks_not_done;
    int id;
    int setup_return;
    setup_return =
        setup(width, height, &display, &win, &gc, &min_color, &max_color);
    if (setup_return != EXIT_SUCCESS) {
        fprintf(stderr, "Unable to initialize display, continuing\n");
    }
    start_time = get_time();
    MPI_Bcast(&min_color, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_color, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    next_row = 0;     
    tasks_not_done = 0;    
    for (int p = 0; p < nworkers; ++p) {
        MPI_Send(&next_row, 1, MPI_INT, p+1, WORK_TAG, MPI_COMM_WORLD);
        ++next_row;
        ++tasks_not_done;
    }
    while (tasks_not_done > 0) {
        MPI_Recv(data_msg, width+1, MPI_LONG, MPI_ANY_SOURCE,
                DATA_TAG, MPI_COMM_WORLD, &status);

        --tasks_not_done;
        id = status.MPI_SOURCE;
        if (next_row < height) {
            MPI_Send(&next_row, 1, MPI_INT, id, WORK_TAG, MPI_COMM_WORLD);
            ++next_row;
            ++tasks_not_done;
        }
        else {
            MPI_Send(&next_row, 0, MPI_INT, id, STOP_TAG, MPI_COMM_WORLD);
        }
        this_row = data_msg[0];
        for (int col = 0; col < width; ++col) {
            if (setup_return == EXIT_SUCCESS) {
                XSetForeground (display, gc, data_msg[col+1]);
                XDrawPoint (display, win, gc, col, this_row);
            }
        }
    }
    if (setup_return == EXIT_SUCCESS) {
        XFlush (display);
    }
    end_time = get_time();
    fprintf(stdout, "\n");
    fprintf(stdout, "MPI program with dynamic task assignment\n");
    fprintf(stdout, "number of worker processes = %d\n", nworkers);
    fprintf(stdout, "center = (%g, %g), size = %g\n",
            (real_max + real_min)/2, (imag_max + imag_min)/2,
            (real_max - real_min)/2);
    fprintf(stdout, "maximum iterations = %d\n", maxiter);
    fprintf(stdout, "execution time in seconds = %g\n", end_time - start_time);
    fprintf(stdout, "\n");
    if (setup_return == EXIT_SUCCESS) {
        interact(display, &win, width, height,
                real_min, real_max, imag_min, imag_max);
    }
    free(data_msg);
    return EXIT_SUCCESS;
}
int worker_pgm(int myID, int width, int height,
        double real_min, double real_max,
        double imag_min, double imag_max,
        int maxiter) {

    MPI_Status status;
    int the_row;
    long min_color, max_color;
    double scale_real, scale_imag, scale_color;
    long *data_msg = malloc((width+1) * sizeof(*data_msg));
    MPI_Bcast(&min_color, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_color, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    scale_real = (double) (real_max - real_min) / (double) width;
    scale_imag = (double) (imag_max - imag_min) / (double) height;
    scale_color = (double) (max_color - min_color) / (double) (maxiter - 1);

    while ( ((MPI_Recv(&the_row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
                        &status)) == MPI_SUCCESS) &&
            (status.MPI_TAG == WORK_TAG) ) {
        data_msg[0] = the_row;

        for (int col = 0; col < width; ++col) {
            complex z, c;
            z.real = z.imag = 0;
            c.real = real_min + ((double) col * scale_real);
            c.imag = imag_min + ((double) (height-1-the_row) * scale_imag);
                                        
            int k = 0;
            double lengthsq, temp;
            do  {
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real + z.imag*z.imag;
                ++k;
            } while (lengthsq < (N*N) && k < maxiter);
            long color = (long) ((k-1) * scale_color) + min_color;
            data_msg[col+1] = color;
        }

        MPI_Send(data_msg, width+1, MPI_LONG, 0, DATA_TAG,
                MPI_COMM_WORLD);
    }

    free(data_msg);
    return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "cuda.h"

#define pixel(i, j, w)  (((j)*(w)) +(i))

int max_num_triangles;
#define B 16

/* A point in 2D space */
struct Point {
    double x;
    double y;
    double value;
};

/* A triangle defined by three points */
struct Triangle {
    struct Point p1;
    struct Point p2;
    struct Point p3;
};

/* Helper function to output the triangles in the Delaunay Triangulation */
void print_triangles(struct Triangle * triangles, int num_triangles) {
    for (int i = 0; i < num_triangles; i++) {
        printf("(%lf, %lf) (%lf, %lf) (%lf, %lf)\n", 
            triangles[i].p1.x, triangles[i].p1.y,
            triangles[i].p2.x, triangles[i].p2.y,
            triangles[i].p3.x, triangles[i].p3.y);   
    }
}

/* Helper function to calculate the distance between two points */
double distance(struct Point * p1, struct Point * p2) {
    double dx = (*p1).x - (*p2).x;
    double dy = (*p1).y - (*p2).y;
    return sqrt(dx*dx + dy*dy);
}

/* Helper function to check if a triangle is clockwise */
int is_ccw(struct Triangle * t) {
    double ax = (*t).p2.x - (*t).p1.x;
    double ay = (*t).p2.y - (*t).p1.y;
    double bx = (*t).p3.x - (*t).p1.x;
    double by = (*t).p3.y - (*t).p1.y;

    double area = ax * by - ay * bx;
    return area > 0;
}

/* Helper function to check if a point is inside a circle defined by three points */
int inside_circle(struct Point * p, struct Triangle * t) {
//      | ax-dx, ay-dy, (ax-dx)² + (ay-dy)² |
//det = | bx-dx, by-dy, (bx-dx)² + (by-dy)² |
//      | cx-dx, cy-dy, (cx-dx)² + (cy-dy)² |

    int clockwise = is_ccw(t);
    
    double ax = (*t).p1.x - (*p).x;
    double ay = (*t).p1.y - (*p).y;
    double bx = (*t).p2.x - (*p).x;
    double by = (*t).p2.y - (*p).y;
    double cx = (*t).p3.x - (*p).x;
    double cy = (*t).p3.y - (*p).y;

    double det = ax*by + bx*cy + cx*ay - ay*bx - by*cx - cy*ax;
    det = (ax*ax + ay*ay) * (bx*cy-cx*by) -
            (bx*bx + by*by) * (ax*cy-cx*ay) +
            (cx*cx + cy*cy) * (ax*by-bx*ay);
    
    if(clockwise)
        return det > 0;
    return det<0;
}

//* Helper function to compute barycentric coordintaes of a point respect a triangle */
void barycentric_coordinates(struct Triangle * t, struct Point * p, double * alpha, double * beta, double * gamma) {
    // Compute the barycentric coordinates of the point with respect to the triangle
    (*alpha) = (((*t).p2.y - (*t).p3.y) * ((*p).x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*p).y - (*t).p3.y)) /
                  (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*beta) = (((*t).p3.y - (*t).p1.y) * ((*p).x - (*t).p3.x) + ((*t).p1.x - (*t).p3.x) * ((*p).y - (*t).p3.y)) /
                 (((*t).p2.y - (*t).p3.y) * ((*t).p1.x - (*t).p3.x) + ((*t).p3.x - (*t).p2.x) * ((*t).p1.y - (*t).p3.y));
    (*alpha) =(*alpha) > 0 ? (*alpha) : 0;
    (*alpha) =(*alpha) < 1 ? (*alpha) : 1;
    (*beta) = (*beta) > 0 ? (*beta) : 0;
    (*beta) = (*beta) < 1 ? (*beta) : 1;
    (*gamma) = 1.0 - (*alpha) - (*beta);
    (*gamma) = (*gamma) > 0 ? (*gamma) : 0;
    (*gamma) = (*gamma) < 1 ? (*gamma) : 1;
}


/* Helper function to check if a point is inside a triangle (IT CAN BE REMOVED)*/
int inside_triangle(struct Triangle * t, struct Point * p) {
    double alpha, beta, gamma;
    barycentric_coordinates(t, p, &alpha, &beta, &gamma); 
    // Check if the barycentric coordinates are positive and add up to 1
    if (alpha > 0 && beta > 0 && gamma > 0) {
        return 1;
    } else {
        return 0;
    }
}

/* Helper function to save an image */   
void save_image(char const * filename, int width, int height, double *image){

   FILE *fp=NULL;
   fp = fopen(filename,"w");
   for(int j=0; j<height; ++j){
      for(int i=0; i<width; ++i){
         fprintf(fp,"%f ", image[pixel(i,j,width)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/* helper function to initialize the points */
void init_points(struct Point* points, int num_points, int width, int height) {
    for(int i = 0; i < num_points; i++) {
        points[i].x =  ((double) rand() / RAND_MAX)*width;
        points[i].y =  ((double) rand() / RAND_MAX)*height;
        points[i].value = 0;//(rand() % 10000) / 100.;
        //printf("Point %d [%f,%f]=%f\n", i, points[i].x, points[i].y, points[i].value);
    }
}

/*Kernel function: to be executed on the device and launched from the host*/
__global__ void count_close_points_CUDA(struct Point* points) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    int j = threadIdx.y + blockIdx.y * blockDim.y; 
    if(i < j){
        if(distance(&points[i], &points[j]) <= 100){
            points[i].value++;                          //stores +1 in value
            points[j].value++; 

        }
    }

}

/*Wraper function to launch the CUDA kernel to count the close points*/
void count_close_points_gpu(struct Point* points, int num_points) {

    //create pointer into the gpu
    struct Point* cudaPoints;

    //allocate memory in the gpu
    cudaMalloc(&cudaPoints, sizeof(points));

    //copy memory into the gpu
    cudaMemcpy(cudaPoints, points, sizeof(points), cudaMemcpyHostToDevice);          //we transfer it from CPU -> GPU

    count_close_points_CUDA<<<num_points, num_points>>>(points);                     //(dimGrid, dimBlock) we want to iterate over every pair
    
    cudaMemcpy(points, cudaPoints, sizeof(points), cudaMemcpyDeviceToHost);          //we transfer it from GPU -> CPU

    //deallocate
    cudaFree(&cudaPoints);

}


// __global__ void delaunay_triangulation_CUDA(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {

//     int nt = 0; 
//     struct Triangle triangle_new;
//     int inside = 0;

//     int i = threadIdx.x / (num_points * num_ponits);
//     int j = (threadIdx.x / num_points) % num_points;
//     int k = threadIdx.x % num_points; 

//     if(i < j && j < k) { //calculate triangle

//         triangle_new.p1 = points[i];
//         triangle_new.p2 = points[j];
//         triangle_new.p3 = points[k];

//         for(int p = 0; p < num_points; p++) {        
//             inside = inside_circle(&points[p], &triangle_new);     // result is 0 or 1 --> need to adapt it to use CUDA
//             if(inside) break;          
//         }

//         //#pragma acc wait                        //waits all previously queued work

//         if(inside == 0) {                       //if no other point is inside the triangle
//             atomicAdd_system(num_triangles, 1)

//             triangles[*num_triangles] = triangle_new;       //nt is updated after the assignation
//         } 



//     }

//     *num_triangles = nt; 
// }

// /*Wraper function to launch the CUDA kernel to compute delaunay triangulation*/
// void delaunay_triangulation_gpu(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {


//     int totalIters = num_points * num_points * num_points; // num_points**3

//     int* d_num_tr; 
//     //cudaMalloc(&d_num_tr, sizeof(int)); //to get back num_triangles

//     int* nt; 
//     cudaMallocManaged(&nt, 4)
//     *nt = 0; 
//     del_Trinag<<<totalIters>>>(points, num_points, triangles, nt); 

//     //cudaMemcpy(num_triangles, d_num_tr, sizeof(int), cudaMemcpyDeviceToHost)

//     //cudaFree(d_num_tr); 

// }


// __global__ void save_triangulation_points_CUDA(struct Point* points) {
    
// }

// /*Wraper function to launch the CUDA kernel to compute delaunay triangulation. 
// Remember to store an image of int's between 0 and 100, where points store 101, and empty areas -1, and points inside triangle the average of value */
// void save_triangulation_image_gpu(struct Point* points, int num_points, struct Triangle* triangles, int num_triangles, int width, int height) {
//     //create structures
//     double* image = (double *) malloc(sizeof(double)*width*height); 



    
//     //write image
//     save_image("image.txt", width, height, image);

//     //free structures
//     free(image);
    
// }

void printCudaInfo() {
    int devNo = 0;
    printf("\n------------------------------------------------------------------\n");
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Maximum grid size is: (");
    for (int i = 0; i < 3; i++)
        printf("%d, ", iProp.maxGridSize[i]);
    printf(")\n");
    printf("Maximum block dim is: (");
    for (int i = 0; i < 3; i++)
        printf("%d, ", iProp.maxThreadsDim[i]);
    printf(")\n");
    printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("------------------------------------------------------------------\n\n");
}

extern "C" int delaunay(int num_points, int width, int height) {
    printCudaInfo();
    
    double start, end;

    max_num_triangles = num_points*30;
    struct Point * points = (struct Point *) malloc(sizeof(struct Point)*num_points);
    struct Triangle * triangles = (struct Triangle *) malloc(sizeof(struct Triangle)*max_num_triangles);
    printf("Maximum allowed number of triangles = %d\n", num_points*30);
    
    init_points(points, num_points, width, height);

    //start = omp_get_wtime();                            //we need to use cudaEvent
    count_close_points_gpu(points, num_points);
    //end = omp_get_wtime();
    
    printf("Counting close points: %f\n", end-start);

    int num_triangles = 0;
    //start = omp_get_wtime();
    //delaunay_triangulation_gpu(points, num_points, triangles, &num_triangles);
    //end = omp_get_wtime();
    printf("Delaunay triangulation: %f\n", end-start);

    printf("Number of generated triangles = %d\n", num_triangles);
    //print_triangles(triangles, num_triangles);

    //start = omp_get_wtime();
    //cudaEventRecord(start);
    //save_triangulation_image_gpu(points, num_points, triangles, num_triangles, width, height);
    //end = omp_get_wtime();
    printf("Generate image: %f\n", end-start);

    //Free memory
    free(points);
    free(triangles);

    return 0;
}
    
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "cuda.h"

#define pixel(i, j, w)  (((j)*(w)) +(i))
#define THREADSPERBLOCK 1024

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


/////////////////////////////////////////////
///
///         CUDA part
///
/////////////////////////////////////////////

__device__ double distance_CUDA(struct Point * p1, struct Point * p2) {
    double dx = (*p1).x - (*p2).x;
    double dy = (*p1).y - (*p2).y;
    return sqrt(dx*dx + dy*dy);
}


/*Kernel function: to be executed on the device and launched from the host*/
__global__ void count_close_points_CUDA(struct Point* points, int num_points) {
    
    int id = threadIdx.x + blockIdx.x * blockDim.x; // get gloval iter

    if(id <= num_points * num_points) return; 

    int i = (id / num_points)%num_points; 
    int j = id%num_points; 

    //printf("Indexes: i:%d and j:%d\n", threadIdx.x, threadIdx.y);
    printf("Indexes: i:%d and j:%d, total iter: %d = %d\n", i, j, id, i * num_points + j);

    if( !(i < j) ) return; 

    double dis = distance_CUDA(&points[i], &points[j]);
    if(dis <= 100){
        points[i].value++;                          //stores +1 in value
        points[j].value++; 
    }
    


}

/*Wraper function to launch the CUDA kernel to count the close points*/
void count_close_points_gpu(struct Point* points, int num_points) {
    int dim_grid = 0;                       //num_blocks
    int dim_block = 0;                      //threads/block

    //create pointer into the gpu
    struct Point* d_points;

    //allocate memory in the gpu
    cudaMalloc(&d_points, sizeof(struct Point) * num_points);

    //copy memory into the gpu
    cudaMemcpy(d_points, points, sizeof(struct Point) * num_points, cudaMemcpyHostToDevice);          //we transfer it from CPU -> GPU

    printf("Num points: %d \n", num_points);
    
    //en una dim
    /*if(num_points <= THREADSPERBLOCK){
        dim_grid = 1;
        dim_block = num_points;
    }else{
        dim_grid = ceil((double)(num_points/1024));             // 32 x 32 = 1024
        dim_block = THREADSPERBLOCK;
    }*/
    /*
    //en 2 dimensiones
    dim_grid = ceil((double)(num_points/THREADSPERBLOCK));             // 32 x 32 = 1024

    dim3 dimGrid((num_points+31)/3);        //?????????????
    dim3 dimBlock(32, 32);

    printf("DimGrid: %d DimBlock: %d \n", dim_grid, dim_block);

    count_close_points_CUDA<<<dimGrid, dimBlock>>>(d_points, num_points);                     //(dimGrid, dimBlock) we want to iterate over every pair
    */

    int n_blocks = (int)ceil(((double)num_points * num_points)/THREADSPERBLOCK); 
    count_close_points_CUDA<<<n_blocks, THREADSPERBLOCK>>>(d_points, num_points);                     //(dimGrid, dimBlock) we want to iterate over every pair


    cudaMemcpy(points, d_points, sizeof(struct Point) * num_points, cudaMemcpyDeviceToHost);          //we transfer it from GPU -> CPU

    //deallocate
    cudaFree(d_points);

}


__global__ void delaunay_triangulation_CUDA(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {

    int nt = 0;                                                 // <- creo que no sirve de nada

    int n_totalIter = threadIdx.x + blockIdx.x * blockDim.x;    //numero de la iteracion global ("entre los 3 fors")

    if(num_points * num_points * num_points < n_totalIter) return; 

    int i = n_totalIter / (num_points * num_points);            //recupera i, j y k
    int j = (n_totalIter / num_points) % num_points;
    int k = n_totalIter % num_points; 

    if( !( i < j && j < k ) ) return; //if the conditions are NOT met, end thread
    //calculate triangle
    struct Triangle triangle_new;

    triangle_new.p1 = points[i];
    triangle_new.p2 = points[j];
    triangle_new.p3 = points[k];
    
    int inside = 0;
    for(int p = 0; p < num_points; p++) {        
        inside = inside_circle(&points[p], &triangle_new);      // result is 0 or 1 --> need to adapt it to use CUDA
        if(inside) break;          
    }
    //#pragma acc wait                                          //waits all previously queued work

    if(inside == 0) {                                           //if no other point is inside the triangle

        triangles[*num_triangles] = triangle_new;               //nt is updated after the assignation
        //aqui podria haver race condition
        atomicAdd_system(num_triangles, 1);                     //atomic add +1

    } 


}

/*Wraper function to launch the CUDA kernel to compute delaunay triangulation*/
void delaunay_triangulation_gpu(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles) {

    /*
    int dim_grid, dim_block;
    dim_grid = 0;
    dim_block = 0;
    */
    struct Point* d_points;                                                                   //ptr GPU
    cudaMalloc(&d_points, sizeof(struct Point) * num_points);                                 //allocate space
    cudaMemcpy(d_points, points, sizeof(struct Point) * num_points, cudaMemcpyHostToDevice);  //data transfer CPU -> GPU

    //repeat for triangles
    struct Triangle* d_triangles;                                                             //ptr GPU
    cudaMalloc(&d_triangles, sizeof(struct Triangle) * num_points * 30);                       //allocate space//in prevois lab, max triangles were num_poits * 30 (or so)
    //no need to copy memory, since the array will be filled there

    int totalIters = num_points * num_points * num_points;                                      // num_points**3

    int* d_nt;                                                                                  //device num triangles
    int h_nt = -3550000;                                                                        // host num triangles //pongo este numero para detectar posibles errores
    cudaMallocManaged(&d_nt, sizeof(int));                                                      //allocate int //lo hago con el managed porque en el tuto lo hacia así. 
    *d_nt = 0;                                                                                  //no sé si esto funciona pero estava en el tuto

    /*
    if(totalIters <= 1024){
        dim_grid = 1;
        dim_block = totalIters;
    }else{
        dim_grid = ceil((double)(totalIters/1024));
        dim_block = 1024;
    }*/

    /*
    dim3 dimGrid(dim_grid);
    dim3 dimBlock(dim_block);
    */
    int n_blocks = (int)ceil(((double)num_points * num_points * num_points)/THREADSPERBLOCK); 

    delaunay_triangulation_CUDA<<<n_blocks, THREADSPERBLOCK>>>(d_points, num_points, d_triangles, d_nt);      //entiendo que falta el block_size(?)
    //Syncronization (?)

    cudaMemcpy(h_nt, d_nt, sizeof(int), cudaMemcpyDeviceToHost);                               ////data transfer GPU -> CPU

    //no need to retrive points since they are not affected
    cudaMemcpy(triangles, d_triangles, sizeof(struct Triangle) * h_nt, cudaMemcpyDeviceToHost); //retrive only the necessary triangles

    *num_triangles = h_nt; //save the value of 
    cudaFree(d_points);
    cudaFree(d_triangles);


    //Display some info and print some of the triangles //delete later
    if(0 <= h_nt){
        int rnd = 5; //bad pseudoransom number generator
        printf("delaunay_triangulation_gpu finalized. Created %d triangles (neg number = error) \n", *num_triangles)
        for(rnd = 5; rnd < *num_triangles; rnd += 20 + (rnd * 73)%29) { //print some of the triangles
            printf("Triangle %d : [(%lf, %lf) (%lf, %lf) (%lf, %lf)] \n", rnd
            triangles[rnd].p1.x, triangles[rnd].p1.y,
            triangles[rnd].p2.x, triangles[rnd].p2.y,
            triangles[rnd].p3.x, triangles[rnd].p3.y);   
        }
    } else printf("There has been an error with d_nt or h_nt variables. h_nt = %d \n", h_nt); 


}


// __global__ void save_triangulation_points_CUDA(struct Point* points, int num_points, struct Triangle* triangles, int* num_triangles, double* image, int width, int height) {
    
//     int i = threadIdx.x + blockIdx.x * blockDim.x; //get position of pixel
//     int j = threadIdx.y + blockIdx.y * blockDim.y; 

//     //declare vars
//     int inside = 0;
//     struct Point pixel, *point;
//     struct Triangle* tr = NULL; 
//     double alpha, beta, gamma;

//     pixel.x = (double)i; 
//     pixel.y = (double)j; 
//     pixel.value = 0;

//     image[pixel(i, j, width)] = -1; //set deafult value


//     for(int k = 0; k < num_triangles; k++){             //recorre todos los triangulos
//         tr = &triangles[k]; 
//         barycentric_coordinates(tr, &pixel, &alpha, &beta, &gamma); 
//         if(0 < alpha && 0 < beta && 0 < gamma){ //if inside triangle
//             image[pixel(i, j, width)] = tr->p1.value * alpha + tr->p2.value * beta + tr->p3.value * gamma;   //sets new value
//             break; // podria ser un return
//         }
//     }



// }

// __global__ void save_BlackBox_CUDA(struct Point* points, int num_points, double* image, int width, int height) {

//     int k = threadIdx.x + blockIdx.x * blockDim.x; 

//     int _x = points[k].x; //get coord of point
//     int _y = points[k].y; 
    
//     int radius = 2; // Total size = (2 * radius + 1)^2

//     //square of size 5

//     for(int i = _x - radius;  i <= _x + radius; i++) { //in a box
//         for(int j = _y - radius; j <= _y + radius; j++) {
//             if(0 <= i && 0 <= j && i < width && j < height) { //if possible
//                 image[(pixel(i, j, width))] = 101.0; //draw black pixel
//             }
//         }
//     }

// }



/*Wraper function to launch the CUDA kernel to compute delaunay triangulation. 
Remember to store an image of int's between 0 and 100, where points store 101, and empty areas -1, and points inside triangle the average of value */

/*
void save_triangulation_image_gpu(struct Point* points, int num_points, struct Triangle* triangles, int num_triangles, int width, int height) {
    
    //create structures
    int size = width * height;
    double* image = (double*) malloc(sizeof(double)*size);

    //copy points to gpu
    struct Point* cudaPoints; //ptr GPU
    cudaMalloc(&cudaPoints, sizeof(struct Point) * num_points); //allocate space
    cudaMemcpy(cudaPoints, points, sizeof(struct Point) * num_points, cudaMemcpyHostToDevice); //data transfer

    //copy triangles to gpu
    struct Triangle* cudaTriangles; //ptr GPU
    cudaMalloc(&cudaTriangles, sizeof(struct Triangle) * num_triangles); //allocate space
    cudaMemcpy(cudaTriangles, triangles, sizeof(struct Triangle) * num_triangles, cudaMemcpyHostToDevice); //data transfer

    double* cudaImage; //ptr GPU
    cudaMalloc(&cudaImage, sizeof(double) * size); //allocate space
    //data created in gpu

    //usamos un thread en la gpu por pixel
    save_triangulation_points_CUDA<<<width, height>>>(cudaPoints, num_points, cudaTriangles, num_triangles, cudaImage, width, height);                                

    cudaFree(cudaTriangles); //not needed anymore

    //wait for next kernel
    //keep image in gpu, no need to move it
    //also keep points there

    save_BlackBox_CUDA<<<num_points>>>(cudaPoints, num_points, cudaImage, width, height); 

    cudaMemcpy(image, cudaImage, sizeof(double) * size, cudaMemcpyDeviceToHost); //retrive image

    cudaFree(cudaPoints); 
    cudaFree(cudaImage); 


    //write image
    save_image("image.txt", width, height, image);

    //free structures
    free(image);
    
}*/

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
    printf("Hi\n");
    count_close_points_gpu(points, num_points);
    //cudaDeviceSynchronize();
    //end = omp_get_wtime();
    
    printf("Counting close points: %f\n", end-start);

    printf("Some points -> point[5]: %f\n", points[5].value);

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
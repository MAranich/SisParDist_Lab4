#include <stdio.h>
#include <stdlib.h>

extern int delaunay(int num_points, int width, int height);

int main(int argc, char* argv[]) {
    int n, w, h;
    
    if(argc == 4) {
        n = atoi(argv[1]);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
    } else {
        fprintf(stderr, "Wrong number of paramethers\nUsage:\n\t%s N W H (for N points in a WxH map)\n",argv[0]);
        return -1;
    }
    
    printf("Delaunay triangulation for %d points in a map with resolution %dx%d\n", n, w, h);
    
    delaunay(n, w, h);

    return 0;
}

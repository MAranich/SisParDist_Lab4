CC=nvcc
CFLAGS=-O3
OBJ=delaunay


gpu:
	$(CC) main.c $(OBJ).cu -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)

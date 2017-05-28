CC=/usr/local/cuda-8.0/bin/nvcc
CFLAGS=-std=c++11 -Xcompiler -fPIC -O3

bfs: $(wildcard *.cu)
	$(CC) -o bfs bfs_basic.cu

clean:
	rm bfs


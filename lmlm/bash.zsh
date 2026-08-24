nvcc -O3 -arch=sm_70 wmma_matrix_mul.cu -o wmma_matrix_mul
./wmma_matrix_mul

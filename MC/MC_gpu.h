#ifndef MC_GPU_H_
#define MC_GPU_H_

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

void solvePagerankByMCGPU(float alpha, int num_m, int threads_per_block, int num_n, int matrix_size, unsigned int seed,
	int* &h_matrix_l, int* &h_accu_row_size, int* &out_deg);
#endif
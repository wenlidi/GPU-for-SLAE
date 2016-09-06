#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <curand.h>
#include <algorithm>
#include<device_launch_parameters.h>
#include <numeric>
#include <time.h>
#include "MC_gpu.h"
#include <curand_kernel.h>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::min;

__global__ void initRNG(curandState *const rngStates,
	const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialise the RNG
	curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ inline void getPoint(float &x, curandState &state)
{
	x = curand_uniform(&state);
}

__global__ void RunMonteCarlo_kernel(int num_n, int m, int* d_matrix_l, int* d_accu_row_size, float alpha, int* d_out_deg, int* d_w, int id, curandState *const rngStates){
	
	unsigned int bid = blockIdx.x;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= m){
		return;
	}

	curandState localState = rngStates[tid];

	int current_state = d_matrix_l[d_accu_row_size[id] + (tid % d_out_deg[id])];
	atomicAdd(d_w + current_state, 1);

	
	float x;
	getPoint(x, localState);
	while (x < 0.85 && d_out_deg[current_state] != 0){
		getPoint(x, localState);
		//TODO
		int random_walk = x * d_out_deg[current_state];
		random_walk = random_walk % d_out_deg[current_state];
		current_state = d_matrix_l[d_accu_row_size[current_state] + random_walk];
		atomicAdd(d_w + current_state, 1);

		getPoint(x, localState);
	}
}

void simulateMC(int device_id, int num_n, int* &d_matrix_l, int* &d_accu_row_size, float alpha, int num_m, unsigned int seed, int threads_per_block, int* &out_deg, int* &d_out_deg){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);
	dim3 dim_grid, dim_block;

	dim_block.x = threads_per_block;

	int *d_w;
	cudaMalloc((void **)&d_w, num_n * sizeof(int));
	cudaMemset(d_w, 0, num_n * sizeof(int));


	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);

	for (int i = 0; i < num_n; i++){
		if (out_deg[i] == 0){
			continue;
		}
		//TODO
		int m = num_m * alpha / out_deg[i];
		m = m * out_deg[i];
		//init
		dim_grid.x = m / dim_block.x;
		if (dim_grid.x == 0) dim_grid.x = 1;
		curandState *d_rngStates;
		cudaMalloc((void **)&d_rngStates, dim_grid.x * dim_block.x * sizeof(curandState));

		initRNG << <dim_grid, dim_block >> >(d_rngStates, seed);
		cudaThreadSynchronize();

		sdkStartTimer(&hTimer);
		//cuda
		RunMonteCarlo_kernel << <dim_grid, dim_block>> >(num_n, m, d_matrix_l, d_accu_row_size, alpha, d_out_deg, d_w, i, d_rngStates);
		
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);
		
		//cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
		//
		cout << "finish" << i << endl;
		cudaFree(d_rngStates);
	}

	vector<unsigned int> results(num_n);
	cudaMemcpy(&results[0], d_w, num_n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	/*
	for (int i = 0; i < num_n; i++){
		cout << results[i] << " ";
	}
	cout << endl;
	*/
	float sum_w = 0;
	for (int i = 0; i< num_n; i++){
		sum_w += results[i] * 1.0 / num_m;
	}
	//float sum_w = static_cast<int>(std::accumulate(results.begin(), results.end(), 0)) * 1.0 / num_m;

	cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl << endl;

	float sum_ans = 0;
	//cout << sum_w << endl;
	for (int i = 0; i < num_n; i++){
		//cout << results[i] << endl;
		cout << (results[i] * 1.0 / num_m + 1) * 1.0 / (sum_w + num_n) * num_n << endl;
	}
	
	/*
	curandStatus_t curandResult;
	curandGenerator_t prng;
	curandResult = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandResult = curandSetPseudoRandomGeneratorSeed(prng, seed);

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);

	//don't know why first time to generate random number need a lot of time 
	curandGenerateUniform(prng, (float *)d_point, 1);

	float generate_random_number_time;
	sdkStartTimer(&hTimer);
	curandGenerateUniform(prng, (float *)d_point, dimension * num_m);
	cudaThreadSynchronize();
	sdkStopTimer(&hTimer);
	generate_random_number_time = sdkGetTimerValue(&hTimer);

	cout << "Elapsed Time of generate random number per variable: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
	sdkResetTimer(&hTimer);

	float last_time = 0;
	float fastest_elapsed_time = 0;
	float slowest_elapsed_time = 0;

	for (int i = 0; i < num_n; i++){
		sdkStartTimer(&hTimer);
		RunMonteCarloGPUForOneVariable_Kernel << <dim_grid, dim_block, dim_block.x * sizeof(float)>> >(d_matrix_p, d_column, d_accu_row_size,
			d_matrix_l_row_sum, f_value, d_point, num_m, dimension, i, d_result);
		reduce_sum_array << <(1, 1, 1), dim_grid, dim_grid.x * sizeof(float) >> >(d_result);
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);

		if (fastest_elapsed_time == 0 || fastest_elapsed_time > sdkGetTimerValue(&hTimer) - last_time){
			fastest_elapsed_time = sdkGetTimerValue(&hTimer) - last_time;
		}
		if (slowest_elapsed_time == 0 || slowest_elapsed_time < sdkGetTimerValue(&hTimer) - last_time){
			slowest_elapsed_time = sdkGetTimerValue(&hTimer) - last_time;
		}
		last_time = sdkGetTimerValue(&hTimer);
		//cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
		cudaMemcpy(&h_exp[i], d_result, sizeof(float), cudaMemcpyDeviceToHost);
		//h_exp[i] = static_cast<float>(std::accumulate(results.begin(), results.end(), 0)) / num_m;
		h_exp[i] = h_exp[i] / num_m;
	//	cout << "finish calculating " << i << "node's expectation" << endl;

		
	}

	curandDestroyGenerator(prng);
	
	cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) + generate_random_number_time * num_n << "ms" << endl;
	cout << "average elapsed time for one variable: " << sdkGetTimerValue(&hTimer) / num_n  + generate_random_number_time<< "ms" << endl;
	cout << "minimal elapsed time for one variable£º" << fastest_elapsed_time + generate_random_number_time << "ms" << endl;
	cout << "maximal elapsed time for one variable£º" << slowest_elapsed_time + generate_random_number_time << "ms" << endl;

	float sum = 0;
	//cout << "ans:" << endl;
	for (int i = 0; i < num_n; i++){
	//	cout << h_exp[i] << " ";
		sum += h_exp[i];
	}
	//cout << endl;
	//cout << sum << endl;
	cout << "total error: " << fabs(sum  / num_n - 1) << endl;
	cout << endl;
	*/
}	

void solvePagerankByMCGPU(float alpha, int num_m, int threads_per_block, int num_n, int matrix_size, unsigned int seed, 
	int* &h_matrix_l, int* &h_accu_row_size, int* &out_deg){
	int device_id = gpuGetMaxGflopsDeviceId();
	cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	cout << "running Metro Carlo method at GPU: ";
	cout << prop.name << endl;

	int *d_matrix_l;
	int *d_accu_row_size;
	int *d_out_deg;

	cudaMalloc((void **)&d_matrix_l, matrix_size * sizeof(int));
	cudaMalloc((void **)&d_accu_row_size, (num_n + 1) * sizeof(int));
	cudaMalloc((void **)&d_out_deg, num_n * sizeof(int));

	cudaMemcpy(d_matrix_l, h_matrix_l, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_accu_row_size, h_accu_row_size, (num_n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_deg, out_deg, num_n * sizeof(int), cudaMemcpyHostToDevice);

	simulateMC(device_id, num_n, d_matrix_l, d_accu_row_size, alpha, num_m, seed, threads_per_block, out_deg, d_out_deg);

	cudaDeviceReset();
}

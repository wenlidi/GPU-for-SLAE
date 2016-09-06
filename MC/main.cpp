#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>

#include "MC_gpu.h"
#include "parameters.h"

using std::string;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;

void printGraphInformation(int num_n, int matrix_size, int* in_deg, int* out_deg){

	int max_out_deg = out_deg[0];
	int min_out_deg = out_deg[0];
	int max_in_deg = in_deg[0];
	int min_in_deg = in_deg[0];
	for (int i = 1; i < num_n; i++){
		if (out_deg[i] > max_out_deg){
			max_out_deg = out_deg[i];
		}
		if (out_deg[i] < min_out_deg){
			min_out_deg = out_deg[i];
		}
		if (in_deg[i] > max_in_deg){
			max_in_deg = in_deg[i];
		}
		if (in_deg[i] < min_in_deg){
			min_in_deg = in_deg[i];
		}
	}
	cout << "Number of nodes:" << num_n << endl;
	cout << "Number of edges:" << matrix_size << endl;
	cout << "Number of maximun out-degree:" << max_out_deg << endl;
	cout << "Number of minimun out-degree:" << min_out_deg << endl;
	cout << "Number of maximun in-degree:" << max_in_deg << endl;
	cout << "Number of minimun in-degree:" << min_in_deg << endl;
	cout << endl;
}

void getInputFromFile(const string& filename, int &num_n, int &matrix_size, 
	int* &h_matrix_l, int* &h_accu_row_size, int* &in_deg, int* &out_deg){

	cout << "reading graph information from " << filename << endl;

	ifstream infile(filename.c_str(), std::ios::in);

	infile >> num_n >> matrix_size;

	out_deg = (int*)malloc(sizeof(int) * num_n);
	in_deg = (int*)malloc(sizeof(int) * num_n);
	h_matrix_l = (int*)malloc(sizeof(float) * matrix_size);
	h_accu_row_size = (int*)malloc(sizeof(int) * (num_n + 1));

	memset(out_deg, 0, sizeof(int) * num_n);
	memset(in_deg, 0, sizeof(int) * num_n);

	int count = 0;
	int last_in = -1;
	int vala, valb;
	for (int i = 0; i < matrix_size; i++){
		infile >> vala >> valb;
		vala -= 1;
		valb -= 1;
		out_deg[vala]++;
		in_deg[valb]++;
		while(last_in != vala){
			h_accu_row_size[count++] = i;
			last_in++;
		}
		h_matrix_l[i] = valb;
	}
	while (count <= num_n){
		h_accu_row_size[count++] = matrix_size;
	}

}

void test(const string& filename, int &num_n, int &matrix_size,
	int* &h_matrix_l, int* &h_accu_row_size, int* &in_deg, int* &out_deg, int* &zero_row, int &zero_row_count){
	cout << filename << endl;
	cout << num_n << endl;
	cout << matrix_size << endl;

	for (int i = 0; i < num_n + 1; i++){
		cout << h_accu_row_size[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < matrix_size; i++){
		cout << h_matrix_l[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < num_n; i++){
		cout << out_deg[i] << " ";
	}
	cout << endl;
}

void setParameters(){
	cout << "reading parameters from parameters.txt..." << endl;

	freopen("parameters.txt", "r", stdin);

	scanf("alpha = %f\n", &alpha);
	scanf("eps = %f\n", &eps);
	scanf("dimension = %d\n", &dimension);
	scanf("num_m = %d\n", &num_m);
	scanf("threads_per_block = %d\n", &threads_per_block);
	scanf("seed = %d\n", &seed);
	fclose(stdin);
	//test

	cout << "alpha = " << alpha << endl;
	cout << "eps = " << eps << endl;
	cout << "d = " << dimension << endl;
	cout << "num_m = " << num_m << endl;
	cout << "threads_per_block = " << threads_per_block << endl;
	cout << "seed = " << seed << endl;
	cout << endl;
}

int main(int argc, char** argv){
	string input_file = "input4.txt";
	string output_file = "output.txt";
	
	if (argc > 1){
		input_file = argv[1];
	}
	if (argc > 2){
		output_file = argv[2];
	}
	freopen(output_file.c_str(), "w", stdout);

	int num_n;
	int matrix_size;
	int* h_matrix_l;
	int* h_accu_row_size;
	int* in_deg;
	int* out_deg;
	int* zero_row;
	int zero_row_count;
	setParameters();

	getInputFromFile(input_file.c_str(), num_n, matrix_size, h_matrix_l, h_accu_row_size, in_deg, out_deg);

	printGraphInformation(num_n, matrix_size, in_deg, out_deg);
	//test(input_file.c_str(), num_n, matrix_size, h_matrix_l, h_accu_row_size, in_deg, out_deg, zero_row, zero_row_count);
	solvePagerankByMCGPU(alpha, num_m, threads_per_block, num_n, matrix_size, seed, h_matrix_l, h_accu_row_size, out_deg);
	fclose(stdout);
	return 0;
}
#include "caffe/util/mm_func.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	template <typename Dtype>
	void __global__ matrix_multiply_kernel(
		const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
// my own matrix multiplication code 		

	double CValue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > M || col > N) return ;
	for(int e = 0; e< K; ++e)
	  CValue += (*(A + e + row * K)) * (*(B + col + e*N));

	*(C + row * K + col) = CValue;
/*		for(int i=0; i < M; i++ )
		  for(int n=0; n < N; n++)
		    for(int j=0; j < K; j++)*/
//			C[i][blockIdx.x] = a[i][threadIdx.x] + b[threadIdx.x][i];
//*(C+ N * i + blockIdx.x) = *(A + K*i + threadIdx.x) * *(B + threadIdx.x * N + i);

//			*(C+N*i+n) = (*(A + K*i + j)) * (*(B + j*N + i));
//*(C + N*i + n) = A[i][j] * B[j][i];
//		C[i][n] =A[i][j]* B[j][i];

	}	
	
	template <typename Dtype>
	void matrix_multiply(const int M, const int N, const int K,
		const Dtype* A, const Dtype* B, Dtype* C)
	{
		matrix_multiply_kernel<Dtype><<<4, 8>>>(M,N,K,A,B,C);
//	matrix_multiply_kernel<Dtype>(M, N, K, A, B, C);
// kernel called here


		// cuBLAS matrix multiplication function
	/*	caffe_gpu_gemm<Dtype>(
			CblasNoTrans, CblasNoTrans, M, N, K,
			(Dtype)1., A, B,(Dtype)0., C);
	*/
	}
	
	template
	void matrix_multiply<float>(
		const int M, const int N, const int K,
		const float* A, const float* B, float* C);
		
	template
	void matrix_multiply<double>(
		const int M, const int N, const int K,
		const double* A, const double* B, double* C);
		
}

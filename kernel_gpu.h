#include <cuda_runtime_api.h>
#include <cuda.h>


extern "C" void cuda_blockmatching(unsigned char *leftCPUImg, unsigned char *rightCPUImg, unsigned char *leftImg, 
		unsigned char *rightImg, float *dispMap, int width, int height, dim3 gridDim, 
		dim3 blockDim, int frame, int delta_min, int delta_max, float borderVal, int steps, int algo);

/*extern "C" void cuda_normalize(float *leftNorm, float *rightNorm, 
		unsigned char *leftImg, unsigned char *rightImg, float medLeft, float medRight, 
		float stdDevLeft, float stdDevRight, int width, int height, 
		dim3 gridDim, dim3 blockDim);*/

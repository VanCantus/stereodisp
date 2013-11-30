#include <iostream>
#include <cuda.h>
#include <stdio.h>

#define MAX(x, y) (x < y) ? y : x
#define MIN(x, y) (x < y) ? x : y

using namespace std;

__device__ float interpolate(const unsigned char left, const unsigned char right, float alpha) {
	return ((float) left * (1.0f - alpha) + (float) right * alpha);
}

__global__ void zmncc(unsigned char *leftImg, unsigned char *rightImg, unsigned char *dispMap, 
		int width, int height, int frame, int delta_min, int delta_max, float ncc_min) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (xOff < frame + 1 + delta_max || xOff >= width - frame - 1 || yOff < frame + 1 || yOff >= height - frame - 1) {
		dispMap[yOff * width + xOff] = 0;
	} else if (xOff < width && yOff < height) {
		int disp = 0;
		float stdDevLeft = 0.0f, medLeft = 0.0f;
		
		for (int delta = -delta_min; delta >= -delta_max; delta--) {
			float valLeft, valRight, cur_ncc = 0.0f, stdDevRight = 0.0f, medRight = 0.0f;
			
			int n = 0;
			for (int j = yOff - frame; j < yOff + frame + 1; j++) {
				for (int i = xOff - frame; i < xOff + frame + 1; i++) {
					if (delta == 0)
						medLeft += leftImg[j * width + i];
					
					medRight += rightImg[j * width + i + delta];
					n++;
				}
			}
			
			if (delta == 0)
				medLeft /= n;
			
			medRight /= n;
			
			for (int j = yOff - frame; j < yOff + frame + 1; j++) {
				for (int i = xOff - frame; i < xOff + frame + 1; i++) {
					valLeft = leftImg[j * width + i];
					valRight = rightImg[j * width + i + delta];
					
					if (delta == 0)
						stdDevLeft += ((valLeft - medLeft) * (valLeft - medLeft)); //pow(valLeft - medLeft, 2.0f);
					
					stdDevRight += ((valRight - medRight) * (valRight - medRight)); //pow(valRight - medRight, 2.0f);
					
					cur_ncc += ((valLeft - medLeft) * (valRight - medRight));
				}
			}
			
			cur_ncc /= sqrt(stdDevLeft * stdDevRight);
			
			if (cur_ncc > ncc_min) {
				ncc_min = cur_ncc;
				disp = delta;
			}
		}
		
		dispMap[yOff * width + xOff] = (unsigned char) (float(-disp - delta_min) / float(delta_max - delta_min) * 255.0f);
	}
}

__global__ void ssd(unsigned char *leftImg, unsigned char *rightImg, unsigned char *dispMap, 
		int width, int height, int frame, int delta_min, int delta_max, float ssd_max, int steps) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (xOff < frame + 1 + delta_max || xOff >= width - frame - 1 || yOff < frame + 1 || yOff >= height - frame - 1) {
		dispMap[yOff * width + xOff] = 0;
	} else if (xOff < width && yOff < height) {
		float disp = delta_min * steps;
		int delta_min_ssd, step_min_ssd;
		float ssd_prev, ssd_right;

		for (int delta = -delta_min + 1; delta >= -delta_max - 1; delta--) {
			for (int s = 0; s < steps; s++) {
				float cur_ssd = 0.0f;

				for (int j = yOff - frame; j < yOff + frame + 1; j++) {
					for (int i = xOff - frame; i < xOff + frame + 1; i++) {
						float tmp_val = float(leftImg[j * width + i] - interpolate(rightImg[j * width + i + delta], 
									rightImg[j * width + i + delta + 1], float(s) / float(steps)));
						cur_ssd += (tmp_val * tmp_val);
					}
				}

				//TODO: minimum berechnen ueber quadr. Funktion
				ssd_prev = cur_ssd;
				
				if (cur_ssd < ssd_max) {
					ssd_max = cur_ssd;
					disp = -delta * steps + s;
				}
			}
		}

		if (disp >= delta_max * steps || disp <= delta_min * steps)
			dispMap[yOff * width + xOff] = 0;
		else
			dispMap[yOff * width + xOff] = (unsigned char) (float(disp - delta_min * steps) 
					/ float(steps * (delta_max - delta_min)) * 255.0f); 
	}
}

__global__ void ncc(unsigned char *leftImg, unsigned char *rightImg, unsigned char *dispMap, 
		int width, int height, int frame, int delta_min, int delta_max, float ncc_min) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (xOff < frame + 1 + delta_max || xOff >= width - frame - 1 || yOff < frame + 1 || yOff >= height - frame - 1) {
		dispMap[yOff * width + xOff] = 0;
	} else if (xOff < width && yOff < height) {
		int disp = 0;
		float valLeft, valRight;

		for (int delta = -delta_min; delta >= -delta_max; delta--) {
			float cur_ncc = 0.0f, lenLeft = 0.0f, lenRight = 0.0f;

			for (int j = yOff - frame; j < yOff + frame + 1; j++) {
				for (int i = xOff - frame; i < xOff + frame + 1; i++) {
					valLeft = leftImg[j * width + i];
					valRight = rightImg[j * width + i + delta];
					
					lenLeft += (valLeft * valLeft);
					lenRight += (valRight * valRight);
					
					cur_ncc += (valLeft * valRight);
				}
			}
			
			cur_ncc /= sqrt(lenLeft * lenRight);
			
			if (cur_ncc > ncc_min) {
				ncc_min = cur_ncc;
				disp = delta;
			}
		}
		
		dispMap[yOff * width + xOff] = (unsigned char) (float(-disp - delta_min) / float(delta_max - delta_min) * 255.0f);
		//dispMap[yOff * width + xOff] = (unsigned char) (float(disp) / float(-delta_max) * 255.0f);
	}
}

__global__ void global_zmncc(unsigned char *leftNorm, unsigned char *rightNorm, unsigned char *dispMap, int width, 
		int height, int frame, int delta_min, int delta_max, float ncc_min, float medLeft, float medRight, 
		float stdDevLeft, float stdDevRight) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;

	if (xOff < frame + 1 + delta_max || xOff >= width - frame - 1 || yOff < frame + 1 || yOff >= height - frame - 1) {
		dispMap[yOff * width + xOff] = 0;
	} else if (xOff < width && yOff < height) {
		int disp = 0;

		for (int delta = -delta_min; delta >= -delta_max; delta--) {
			float cur_ncc = 0.0f;
			
			for (int j = yOff - frame; j < yOff + frame + 1; j++) {
				if (j < 0 || j >= height)
					break;
				
				for (int i = xOff - frame; i < xOff + frame + 1; i++) {
					if (i + delta >= width || i + delta < 0 || i < 0 || i >= width) 
						break;

					cur_ncc += ((leftNorm[j * width + i] - medLeft) * (rightNorm[j * width + i + delta] - medRight));
				}
			}
			
			cur_ncc /= (stdDevLeft * stdDevRight);

			if (cur_ncc > ncc_min) {
				ncc_min = cur_ncc;
				
				disp = delta;
			}
		}
		
		dispMap[yOff * width + xOff] = (unsigned char) (float(-disp - delta_min) / float(delta_max - delta_min) * 255.0f);
		//dispMap[yOff * width + xOff] = (unsigned char) (float(disp) / float(-delta_max) * 255.0f);
	}
}

__global__ void normalized_ssd(unsigned char *leftNorm, unsigned char *rightNorm, unsigned char *dispMap, int width, 
		int height, int frame, int delta_min, int delta_max, float ssd_min, float medLeft, float medRight, 
		float stdDevLeft, float stdDevRight) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (xOff < frame + 1 + delta_max || xOff >= width - frame - 1 || yOff < frame + 1 || yOff >= height - frame - 1) {
		dispMap[yOff * width + xOff] = 0;
	} else if (xOff < width && yOff < height) {
		int disp = 0;
		
		for (int delta = -delta_min; delta >= -delta_max; delta--) {
			float cur_ssd = 0.0f;
			
			for (int j = yOff - frame; j < yOff + frame + 1; j++) {
				if (j < 0 || j >= height)
					break;
				
				for (int i = xOff - frame; i < xOff + frame + 1; i++) {
					if (i + delta >= width || i + delta < 0 || i < 0 || i >= width) 
						break;
					
					cur_ssd += pow((leftNorm[j * width + i] - medLeft) - (rightNorm[j * width + i + delta] - medRight), 2.0f);
				}
			}
			
			cur_ssd /= (stdDevLeft * stdDevRight);
			
			if (cur_ssd < ssd_min) {
				ssd_min = cur_ssd;
				
				disp = delta;
			}
		}
		
		dispMap[yOff * width + xOff] = (unsigned char) (float(-disp - delta_min) / float(delta_max - delta_min) * 255.0f);
		//dispMap[yOff * width + xOff] = (unsigned char) (float(disp) / float(-delta_max) * 255.0f);
	}
}

__global__ void normalize(float *leftNorm, float *rightNorm, unsigned char *leftImg, 
		unsigned char *rightImg, float medLeft, float medRight, float stdDevLeft, 
		float stdDevRight, int width, int height) {
	int xOff = blockIdx.x * blockDim.x + threadIdx.x;
	int yOff = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (xOff < width && yOff < height) {
		leftNorm[yOff * width + xOff] = (float) (0.5f * ((float(leftImg[yOff * width + xOff]) 
				- medLeft) / stdDevLeft + 1.0f));
		rightNorm[yOff * width + xOff] = (float) (0.5f * ((float(rightImg[yOff * width + xOff]) 
				- medRight) / stdDevRight + 1.0f));
	}
}

void calcMed(unsigned char *leftCPUImg, unsigned char *rightCPUImg, int width, int height, 
		float *medLeft, float *medRight) {
	float tmpMedLeft = 0.0f, tmpMedRight = 0.0f;
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			tmpMedLeft += leftCPUImg[i * width + j];
			tmpMedRight += rightCPUImg[i * width + j];
		}
	}
	
	*medLeft = tmpMedLeft / (float) (width * height);
	*medRight = tmpMedRight / (float) (width * height);
}

void calcStdDev(unsigned char *leftCPUImg, unsigned char *rightCPUImg, int width, int height, 
		float medLeft, float medRight, float *stdDevLeft, float *stdDevRight) {
	float tmpStdDevLeft = 0.0f, tmpStdDevRight = 0.0f;
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			tmpStdDevLeft += pow(leftCPUImg[i * width + j] - medLeft, 2.0);
			tmpStdDevRight += pow(rightCPUImg[i * width + j] - medRight, 2.0);
		}
	}
	
	*stdDevLeft = sqrt(tmpStdDevLeft);
	*stdDevRight = sqrt(tmpStdDevRight);
}

extern "C" void cuda_blockmatching(unsigned char *leftCPUImg, unsigned char *rightCPUImg, unsigned char *leftImg, 
		unsigned char *rightImg, unsigned char *dispMap, int width, int height, dim3 gridDim, 
		dim3 blockDim, int frame, int delta_min, int delta_max, float borderVal, int steps, int algo) {
	if (algo == 0) { //SSD
		ssd<<<gridDim, blockDim>>>(leftImg, rightImg, dispMap, width, height, 
				frame, delta_min, delta_max, borderVal, steps);
	} else if (algo == 1) { //normalized SSD
		float medLeft, medRight, stdDevLeft, stdDevRight;
		float *leftNorm, *rightNorm;
		
		if (cudaMalloc((void **) (&leftNorm), width * height * sizeof(float)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc of left normalized image" << endl;
			exit(EXIT_FAILURE);
		}
		if (cudaMalloc((void **) (&rightNorm), width * height * sizeof(float)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc of right normalized image" << endl;
			exit(EXIT_FAILURE);
		}
		
		calcMed(leftCPUImg, rightCPUImg, width, height, &medLeft, &medRight);
		calcStdDev(leftCPUImg, rightCPUImg, width, height, medLeft, medRight, &stdDevLeft, &stdDevRight);
		
		/*normalize<<<gridDim, blockDim>>>(leftNorm, rightNorm, leftImg, rightImg, 
				medLeft, medRight, stdDevLeft, stdDevRight, width, height);
		
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			cerr << "Normalize: Error in kernel execution: " 
					<< cudaGetErrorString(err) << endl;
			exit(EXIT_FAILURE);
		}
	
		if (cudaDeviceSynchronize() != cudaSuccess) {
			cerr << "Normalize: Error in kernel execution: " 
					<< cudaGetErrorString(cudaGetLastError()) << endl;
			exit(EXIT_FAILURE);
		}*/
		
		normalized_ssd<<<gridDim, blockDim>>>(leftImg, rightImg, dispMap, width, 
				height, frame, delta_min, delta_max, borderVal, medLeft, medRight, stdDevLeft, stdDevRight);
	} else if (algo == 2) { //local ZMNCC
		ncc<<<gridDim, blockDim>>>(leftImg, rightImg, dispMap, width, height, 
				frame, delta_min, delta_max, borderVal);
	} else if (algo == 3) { //local ZMNCC
		zmncc<<<gridDim, blockDim>>>(leftImg, rightImg, dispMap, width, height, 
				frame, delta_min, delta_max, borderVal);
	} else { //global ZMNCC
		float medLeft, medRight, stdDevLeft, stdDevRight;
		float *leftNorm, *rightNorm;
		
		if (cudaMalloc((void **) (&leftNorm), width * height * sizeof(float)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc of left normalized image" << endl;
			exit(EXIT_FAILURE);
		}
		if (cudaMalloc((void **) (&rightNorm), width * height * sizeof(float)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc of right normalized image" << endl;
			exit(EXIT_FAILURE);
		}
		
		calcMed(leftCPUImg, rightCPUImg, width, height, &medLeft, &medRight);
		calcStdDev(leftCPUImg, rightCPUImg, width, height, medLeft, medRight, &stdDevLeft, &stdDevRight);
		
		/*normalize<<<gridDim, blockDim>>>(leftNorm, rightNorm, leftImg, rightImg, 
				medLeft, medRight, stdDevLeft, stdDevRight, width, height);
		
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			cerr << "Normalize: Error in kernel execution: " 
					<< cudaGetErrorString(err) << endl;
			exit(EXIT_FAILURE);
		}
	
		if (cudaDeviceSynchronize() != cudaSuccess) {
			cerr << "Normalize: Error in kernel execution: " 
					<< cudaGetErrorString(cudaGetLastError()) << endl;
			exit(EXIT_FAILURE);
		}*/
		
		global_zmncc<<<gridDim, blockDim>>>(leftImg, rightImg, dispMap, width, 
				height, frame, delta_min, delta_max, borderVal, medLeft, medRight, stdDevLeft, stdDevRight);
	}
	
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		cerr << "Error in kernel execution: " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
	
	if (cudaDeviceSynchronize() != cudaSuccess) {
		cerr << "Error in kernel execution: " << cudaGetErrorString(cudaGetLastError()) << endl;
		exit(EXIT_FAILURE);
	}
}


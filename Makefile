CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda#/soft/devtools/nvidia/cuda-5.5
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include $(shell pkg-config --cflags opencv) #-I/usr/include/opencv2
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 $(shell pkg-config --libs opencv) -lcudart

all:
	$(CXX) $(CFLAGS) -c main.cpp -o Debug/main.o
	nvcc $(CUDAFLAGS)-c kernel_gpu.cu -o Debug/kernel_gpu.o
	$(CXX) Debug/main.o Debug/kernel_gpu.o -o blockmatching $(LDFLAGS)

clean:
	rm -f Debug/*.o blockmatching


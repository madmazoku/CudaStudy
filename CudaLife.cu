#include "CudaLife.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cudaDecKernel(
	cudaSurfaceObject_t surfaceIn,
	cudaSurfaceObject_t surfaceOut,
	unsigned int width,
	unsigned int height
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 cell;
	surf2Dread(&cell, surfaceIn, x * 4, y);
	if (cell.x != 0 && cell.x != 0x80)
		cell = make_uchar4(cell.x - 1, cell.x - 1, cell.x - 1, 0xff);
	surf2Dwrite(cell, surfaceOut, x * 4, y);
}

__global__ void cudaLifeKernel(
	cudaSurfaceObject_t surfaceIn,
	cudaSurfaceObject_t surfaceOut,
	unsigned int width,
	unsigned int height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int neighbors = 0;
	if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
		for (int yo = -1; yo <= 1; ++yo)
			for (int xo = -1; xo <= 1; ++xo)
				if (yo != 0 || xo != 0) {
					int ix = x + xo;
					int iy = y + yo;
					if (ix < 0) ix += width;
					if (ix >= width) ix -= width;
					if (iy < 0) iy += height;
					if (iy >= height) iy -= height;

					uchar4 data;
					surf2Dread(&data, surfaceIn, ix * 4, iy);
					if (data.x > 0x7f)
						++neighbors;
				}
	}
	else
	{
		for (int yo = -1; yo <= 1; ++yo)
			for (int xo = -1; xo <= 1; ++xo)
				if (yo != 0 || xo != 0) {
					int ix = x + xo;
					int iy = y + yo;
					uchar4 data;
					surf2Dread(&data, surfaceIn, ix * 4, iy);
					if (data.x > 0x7f)
						++neighbors;
				}
	}
	uchar4 cell;
	surf2Dread(&cell, surfaceIn, x * 4, y);

	uchar4 data;
	if (neighbors == 2)
		data = make_uchar4(cell.x, cell.x, cell.x, 0xff);
	else if (neighbors == 3)
		if (cell.x > 0x7f)
			data = make_uchar4(cell.x, cell.x, cell.x, 0xff);
		else
			data = make_uchar4(0xff, 0xff, 0xff, 0xff);
	else
		if (cell.x > 0x7f)
			data = make_uchar4(0x7f, 0x7f, 0x7f, 0xff);
		else
			data = make_uchar4(cell.x, cell.x, cell.x, 0xff);

	surf2Dwrite(data, surfaceOut, x * 4, y);
}

extern "C" void doDecKernel(cudaSurfaceObject_t surfaceIn, cudaSurfaceObject_t surfaceOut, unsigned int width, unsigned int height) {
	dim3 threads(32, 32);
	dim3 blocks(width / threads.x, height / threads.y);
	if (blocks.x * threads.x < width) ++blocks.x;
	if (blocks.y * threads.y < height) ++blocks.y;

	cudaDecKernel << <blocks, threads >> > (surfaceIn, surfaceOut, width, height);
}

extern "C" void doLifeKernel(cudaSurfaceObject_t surfaceIn, cudaSurfaceObject_t surfaceOut, unsigned int width, unsigned int height) {
	dim3 threads(32, 32);
	dim3 blocks(width / threads.x, height / threads.y);
	if (blocks.x * threads.x < width) ++blocks.x;
	if (blocks.y * threads.y < height) ++blocks.y;

	cudaLifeKernel << <blocks, threads >> > (surfaceIn, surfaceOut, width, height);
}
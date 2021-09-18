#include "CudaLife.cuh"

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


constexpr sf::Uint32 getMatrixWeightsSum(const sf::Uint8* pMatrix, unsigned int width, unsigned int height) {
	const sf::Uint8* pMatrixPos = pMatrix;
	const sf::Uint8* pMatrixEnd = pMatrix + width * height;

	sf::Uint32 sum = 0;
	while (pMatrixPos < pMatrixEnd)
		sum += *(pMatrixPos++);

	return sum;
}

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
		if(cell.x > 0x7f)
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

CudaLife::CudaLife(unsigned int width, unsigned int height)
	:  _width(width), _height(height), _pTextureIn(nullptr), _pTextureOut(nullptr)
{

}

void CudaLife::initialize() {
	_pTextureIn = std::make_shared<CudaTexture>(_width, _height);
	_pTextureOut = std::make_shared<CudaTexture>(_width, _height);

	_pTextureIn->registerResource();
	_pTextureOut->registerResource();
}

void CudaLife::update() {
	//std::swap(_pTextureIn, _pTextureOut);

	_pTextureIn->mapResource();
	_pTextureOut->mapResource();

	cudaArray* arrayIn = _pTextureIn->mappedArray();
	cudaArray* arrayOut = _pTextureOut->mappedArray();

	_pTextureIn->createSurface();
	_pTextureOut->createSurface();

	cudaErrorCheck(cudaGetLastError());

	dim3 threads(32, 32);
	dim3 blocks(_width / threads.x, _height / threads.y);
	if (blocks.x * threads.x < _width) ++blocks.x;
	if (blocks.y * threads.y < _height) ++blocks.y;
	cudaDecKernel<< <blocks, threads >> > (
		_pTextureOut->surface(),
		_pTextureIn->surface(),
		_width,
		_height
		);
	cudaLifeKernel << <blocks, threads >> > (
		_pTextureIn->surface(),
		_pTextureOut->surface(),
		_width,
		_height
	);

	cudaErrorCheck(cudaPeekAtLastError());
	cudaErrorCheck(cudaDeviceSynchronize());

	_pTextureOut->destroySurface();
	_pTextureIn->destroySurface();

	_pTextureOut->unmapResource();
	_pTextureIn->unmapResource();
}

void CudaLife::shutdown() {
	_pTextureOut->unregisterResource();
	_pTextureIn->unregisterResource();

	_pTextureIn = nullptr;
	_pTextureOut = nullptr;
}
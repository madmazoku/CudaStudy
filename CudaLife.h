#pragma once

#include <memory>

#include "CudaTexture.h"

#include "CudaLife.cuh"

class CudaLife
{
protected:
	unsigned int _width;
	unsigned int _height;

	std::shared_ptr<CudaTexture> _pTextureIn;
	std::shared_ptr<CudaTexture> _pTextureOut;

public:
	CudaLife(unsigned int width, unsigned int height)
		: _width(width), _height(height), _pTextureIn(nullptr), _pTextureOut(nullptr)
	{}

	void initialize() {
		_pTextureIn = std::make_shared<CudaTexture>(_width, _height);
		_pTextureOut = std::make_shared<CudaTexture>(_width, _height);

		_pTextureIn->registerResource();
		_pTextureOut->registerResource();
	}

	void update() {
		//std::swap(_pTextureIn, _pTextureOut);

		_pTextureIn->mapResource();
		_pTextureOut->mapResource();

		cudaArray* arrayIn = _pTextureIn->mappedArray();
		cudaArray* arrayOut = _pTextureOut->mappedArray();

		_pTextureIn->createSurface();
		_pTextureOut->createSurface();

		cudaErrorCheck(cudaGetLastError());

		doDecKernel(_pTextureOut->surface(), _pTextureIn->surface(), _width, _height);
		doLifeKernel(_pTextureIn->surface(), _pTextureOut->surface(), _width, _height);

		cudaErrorCheck(cudaPeekAtLastError());
		cudaErrorCheck(cudaDeviceSynchronize());

		_pTextureOut->destroySurface();
		_pTextureIn->destroySurface();

		_pTextureOut->unmapResource();
		_pTextureIn->unmapResource();
	}

	void shutdown() {
		_pTextureOut->unregisterResource();
		_pTextureIn->unregisterResource();

		_pTextureIn = nullptr;
		_pTextureOut = nullptr;
	}

	CudaTexture& inputTexture() { return *_pTextureIn; }
	CudaTexture& outputTexture() { return *_pTextureOut; }
};

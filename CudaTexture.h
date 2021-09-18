#pragma once

#include <Windows.h>
#include <cuda_gl_interop.h>

#include <SFML/Graphics.hpp>

#include "utils.h"

class CudaTexture {
protected:
	sf::Texture _texture;
	sf::Sprite _sprite;
	cudaGraphicsResource_t _resource;
	cudaSurfaceObject_t _surface;

public:
	CudaTexture(unsigned int width, unsigned int height) : _resource(nullptr), _surface(0) {
		if (!_texture.create(width, height))
			throw std::runtime_error("Can't create texture");
		_sprite.setTexture(_texture, true);
	}

	void registerResource() {
		cudaErrorCheck(cudaGraphicsGLRegisterImage(&_resource, _texture.getNativeHandle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	}
	void unregisterResource() {
		cudaErrorCheck(cudaGraphicsUnregisterResource(_resource));
	}

	void mapResource(cudaStream_t stream = 0) {
		cudaErrorCheck(cudaGraphicsMapResources(1, &_resource, stream));
	}
	void unmapResource(cudaStream_t stream = 0) {
		cudaErrorCheck(cudaGraphicsUnmapResources(1, &_resource, stream));
	}

	void createSurface(unsigned int arrayIndex = 0, unsigned int mipLevel = 0) {
		cudaResourceDesc resourceDesc;
		memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
		resourceDesc.resType = cudaResourceTypeArray;
		resourceDesc.res.array.array = mappedArray(arrayIndex, mipLevel);

		cudaErrorCheck(cudaCreateSurfaceObject(&_surface, &resourceDesc));
	}
	void destroySurface() {
		cudaErrorCheck(cudaDestroySurfaceObject(_surface));
	}

	cudaArray_t mappedArray(unsigned int arrayIndex = 0, unsigned int mipLevel = 0) {
		cudaArray_t textureArray;
		cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&textureArray, _resource, arrayIndex, mipLevel));
		return textureArray;
	}

	sf::Texture& texture() { return _texture; }
	sf::Sprite& sprite() { return _sprite; }
	cudaGraphicsResource_t resource() { return _resource; }
	cudaSurfaceObject_t surface() { return _surface; }
};

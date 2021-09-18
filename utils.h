#pragma once

#include <sstream>
#include <iostream>
#include <stdexcept>

#define cudaErrorCheck(x) cudaErrorCheckImpl(x, __FILE__, __LINE__)

inline void cudaErrorCheckImpl(cudaError_t cudaError, const char* file, long line) {
	if (cudaError != cudaSuccess) {
		std::stringstream ss;
		ss << "[" << file << ":" << line << "] cudaError [" << cudaError << "]: " << cudaGetErrorString(cudaError);
		std::cerr << ss.str() << std::endl;
		throw std::runtime_error(ss.str());
	}
}

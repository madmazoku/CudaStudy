#pragma once

extern "C" void doDecKernel(cudaSurfaceObject_t surfaceIn, cudaSurfaceObject_t surfaceOut, unsigned int width, unsigned int height);
extern "C" void doLifeKernel(cudaSurfaceObject_t surfaceIn, cudaSurfaceObject_t surfaceOut, unsigned int width, unsigned int height);
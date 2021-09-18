#include <random>
#include <iostream>
#include <thread>
#include <mutex>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "CudaLife.cuh"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main_1() {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

inline int warpCoord(int value, int limit) {
    if (value < 0)
        return value + limit;
    else if (value >= limit)
        return value - limit;
    else
        return value;
}

constexpr sf::Uint32 getMatrixWeight(const sf::Uint8* pMatrix, unsigned int width, unsigned int height) {
    const sf::Uint8* pMatrixPos = pMatrix;
    const sf::Uint8* pMatrixEnd = pMatrix + width * height;

    sf::Uint32 sum = 0;
    while (pMatrixPos < pMatrixEnd)
        sum += *(pMatrixPos++);

    return sum;
}

const sf::Uint8 sumMatrix[] = {
    0, 1, 1, 1, 0,
    1, 1, 2, 1, 1,
    1, 2, 3, 2, 1,
    1, 1, 2, 1, 1,
    0, 1, 1, 1, 0,
};
const sf::Uint32 sumMatrixWeight = getMatrixWeight(sumMatrix, 5, 5);

void updateTexture(sf::Uint8* pBoardOut, sf::Uint8* pBoardIn, sf::Uint32* pPixels, sf::Texture& texture) {
    sf::Vector2u size = texture.getSize();
    for (unsigned int y = 0; y < size.y; ++y)
        for (unsigned int x = 0; x < size.x; ++x) {
            sf::Uint32 sum = 0;
            for (int sy = -2; sy <= 2; ++sy) {
                int ry = warpCoord(y + sy, size.y);
                for (int sx = -2; sx <= 2; ++sx) {
                    int rx = warpCoord(x + sx, size.x);
                    sum += pBoardIn[ry * size.x + rx] * sumMatrix[(sy + 2) * 5 + (sx + 2)];
                }
            }
            pBoardOut[y * size.x + x] = sum / sumMatrixWeight;
        }

    sf::Uint8* pBoardOutStart = pBoardOut;
    sf::Uint32* pPixelsPos = pPixels;
    sf::Uint32* pPixelsEnd = pPixels+ size.x * size.y;
    while (pPixelsPos < pPixelsEnd) {
        sf::Uint8 cell = *(pBoardOutStart++);
        *(pPixelsPos++) = 0xff000000 | (sf::Uint32(cell)) | (sf::Uint32(cell) << 8) | (sf::Uint32(cell) << 16);
    }
    texture.update(reinterpret_cast<sf::Uint8*>(pPixels));
}

int main_2()
{
    sf::VideoMode vmDesktop = sf::VideoMode::getDesktopMode();
    sf::VideoMode vmWindow(vmDesktop.width >> 1, vmDesktop.height >> 1);
    sf::RenderWindow window(vmWindow, "SFML works!");

    long boardSize = vmDesktop.width * vmDesktop.height;
    sf::Uint8* pBoards[2];
    pBoards[0] = new sf::Uint8[boardSize];
    pBoards[1] = new sf::Uint8[boardSize];
    sf::Uint32* pPixels = new sf::Uint32[boardSize];

    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<int> uniform_dist(0, std::numeric_limits<sf::Uint8>::max());
    sf::Uint8* pBoard = pBoards[0];
    sf::Uint8* pBoardEnd = pBoard + boardSize;
    while (pBoard < pBoardEnd)
        *(pBoard++) = uniform_dist(re);

    sf::Texture texture;
    if (!texture.create(vmWindow.width, vmWindow.height))
        return -1;
    sf::Sprite sprite(texture);

    sf::Clock clock;
    sf::Time lastTime = clock.getElapsedTime();
    long frameCount = 0;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        std::swap(pBoards[0], pBoards[1]);
        updateTexture(pBoards[0], pBoards[1], pPixels, texture);

        window.clear();
        window.draw(sprite);
        window.display();

        ++frameCount;
        sf::Time currentTime = clock.getElapsedTime();
        float timeDiffSecond = currentTime.asSeconds() - lastTime.asSeconds();
        if (timeDiffSecond >= 1.0f && frameCount >= 10) {
            float fps = float(frameCount) / timeDiffSecond;
            std::cout << "FPS: " << floor(fps) << std::endl;
            lastTime = currentTime;
            frameCount = 0;
        }
    }
}

int main() {
    sf::VideoMode vmDesktop = sf::VideoMode::getDesktopMode();

    int deviceCount;
    int deviceSelected = 0;
    cudaErrorCheck(cudaGetDeviceCount(&deviceCount));

    std::cout << "CUDA devices [" << deviceCount << "]" << std::endl;
    for (int device = 0; device < deviceCount; ++device) {
        std::cout << "\t# " << device << std::endl;
        cudaDeviceProp deviceProp;
        cudaErrorCheck(cudaGetDeviceProperties(&deviceProp, device));
        std::cout << "\t\tDevice Name:             " << deviceProp.name << std::endl;
        std::cout << "\t\tMemory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "\t\tMemory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "\t\tWarp size (threads):     " << deviceProp.warpSize << std::endl;
        std::cout << "\t\tProcessor count:         " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "\t\tIs integrated:           " << deviceProp.integrated << std::endl;
        std::cout << "\t\tCompute mode:            " << deviceProp.computeMode << std::endl;
        std::cout << "\t\tIs multiGPU board:       " << deviceProp.isMultiGpuBoard << std::endl;
        deviceSelected = device;
    }

    cudaErrorCheck(cudaSetDevice(deviceSelected));

#ifdef _DEBUG
    sf::VideoMode vmWindow(vmDesktop.width >> 1, vmDesktop.height >> 1);
    sf::RenderWindow window(vmWindow, "SFML works!");
#else
    sf::VideoMode vmWindow(vmDesktop.width, vmDesktop.height);
    sf::RenderWindow window(vmWindow, "SFML works!", sf::Style::None);
#endif // _DEBUG

    window.setVerticalSyncEnabled(false);

    sf::ContextSettings settings = window.getSettings();

    std::cout << "OpenGL setup:" << std::endl;
    std::cout << "\tdepth bits:" << settings.depthBits << std::endl;
    std::cout << "\tstencil bits:" << settings.stencilBits << std::endl;
    std::cout << "\tantialiasing level:" << settings.antialiasingLevel << std::endl;
    std::cout << "\tversion:" << settings.majorVersion << "." << settings.minorVersion << std::endl;

    std::shared_ptr<CudaLife> pCS = std::make_shared<CudaLife>(vmWindow.width, vmWindow.height);
    pCS->initialize();

    sf::Uint8* pBoard = new sf::Uint8[vmWindow.width * vmWindow.height * 4];

    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_int_distribution<int> uniform_dist(0, 10);
    sf::Uint32* pBoardPos = reinterpret_cast<sf::Uint32*>(pBoard);
    sf::Uint32* pBoardEnd = pBoardPos + vmWindow.width * vmWindow.height;
    while (pBoardPos < pBoardEnd) {
        sf::Uint32 cell = uniform_dist(re) == 0 ? 0xff : 0x00;
        *(pBoardPos++) = 0xff000000 | (cell) | (cell << 8) | (cell << 16);
    }
    pCS->outputTexture().texture().update(pBoard);

    sf::Clock clock;
    sf::Time lastTime = clock.getElapsedTime();
    long frameCount = 0;

    while (window.isOpen())
    {
        window.clear();
        window.draw(pCS->outputTexture().sprite());
        window.display();

        pCS->update();

        ++frameCount;
        sf::Time currentTime = clock.getElapsedTime();
        float timeDiffSecond = currentTime.asSeconds() - lastTime.asSeconds();
        if (timeDiffSecond >= 1.0f) {
            float fps = float(frameCount) / timeDiffSecond;
            std::cout << "FPS: " << std::floor(fps) << std::endl;
            lastTime = currentTime;
            frameCount = 0;
        }

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                pCS->shutdown();
                window.close();
            }
        }
    }

    delete[] pBoard;

    pCS = nullptr;
}
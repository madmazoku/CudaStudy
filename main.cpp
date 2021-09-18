#include <random>
#include <iostream>

#include <cuda_runtime.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "utils.h"
#include "CudaLife.h"

void fillRandomBoard(sf::Texture& texture) {
	sf::Vector2u size = texture.getSize();
	unsigned long boardSize = size.x * size.y;
	sf::Uint8* pBoard = new sf::Uint8[boardSize * 4];

	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_int_distribution<int> uniform_dist(0, 10);
	sf::Uint32* pBoardPos = reinterpret_cast<sf::Uint32*>(pBoard);
	sf::Uint32* pBoardEnd = pBoardPos + boardSize;
	while (pBoardPos < pBoardEnd) {
		sf::Uint32 cell = uniform_dist(re) == 0 ? 0xff : 0x00;
		*(pBoardPos++) = 0xff000000 | (cell) | (cell << 8) | (cell << 16);
	}
	texture.update(pBoard);

	delete[] pBoard;
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

	std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "OpenGL setup:" << std::endl;
	std::cout << "\tdepth bits:" << settings.depthBits << std::endl;
	std::cout << "\tstencil bits:" << settings.stencilBits << std::endl;
	std::cout << "\tantialiasing level:" << settings.antialiasingLevel << std::endl;
	std::cout << "\tversion:" << settings.majorVersion << "." << settings.minorVersion << std::endl;

	std::shared_ptr<CudaLife> pCS = std::make_shared<CudaLife>(vmWindow.width, vmWindow.height);
	pCS->initialize();

	fillRandomBoard(pCS->outputTexture().texture());

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

			sf::Image imageIn = pCS->inputTexture().texture().copyToImage();
			sf::Image imageOut = pCS->outputTexture().texture().copyToImage();
			const sf::Uint32* pBoardInPos = reinterpret_cast<const sf::Uint32*>(imageIn.getPixelsPtr());
			const sf::Uint32* pBoardOutPos = reinterpret_cast<const sf::Uint32*>(imageOut.getPixelsPtr());
			const sf::Uint32* pBoardInEnd = pBoardInPos + vmWindow.width * vmWindow.height;
			long diffCells = 0;
			long emptyCells = 0;
			while (pBoardInPos < pBoardInEnd)
				if (((*(pBoardInPos++) & 0xff) > 0x7f) != ((*(pBoardOutPos++) & 0xff) > 0x7f))
					diffCells++;
			float diffCellsPrc = diffCells * 100.0f / (vmWindow.width * vmWindow.height);
			std::cout << "different cells: " << diffCells << " [ " << diffCellsPrc << "% ]" << std::endl;
			if (diffCellsPrc < 0.9)
				fillRandomBoard(pCS->outputTexture().texture());
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

	pCS = nullptr;
}
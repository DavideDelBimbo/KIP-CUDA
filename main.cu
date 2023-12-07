#include <iostream>
#include <chrono>

#include "image.h"
#include "kernel.h"
#include "./parallel/convolution.h"
#include "./sequential/convolution.h"


int main() {
    Image image("images/4K_grey.jpg", 1, true);    
    Kernel kernel = Kernel::edge_detection_kernel();

    std::cout << kernel << std::endl;

    /*auto t1 = std::chrono::high_resolution_clock::now();
    Image s = Sequential::Convolution::convolve(image, kernel, PaddingType::MIRROR);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Convolution sequential: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    //s.save_image("images/sequential.png");*/

    /*auto t1 = std::chrono::high_resolution_clock::now();
    Image a = Parallel::Convolution::convolve_global(image, kernel, PaddingType::MIRROR);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Convolution global: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    //a.save_image("images/global.png");*/

    auto t1 = std::chrono::high_resolution_clock::now();
    Image b = Parallel::Convolution::convolve_constant(image, kernel, PaddingType::MIRROR);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Convolution constant: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    //b.save_image("images/constant.png");

    /*auto t1 = std::chrono::high_resolution_clock::now();
    Image c = Parallel::Convolution::convolve_shared(image, kernel, PaddingType::MIRROR);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Convolution shared: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());*/

    //c.save_image("images/shared.png");

    return 0;
}
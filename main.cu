#include <iostream>
#include <chrono>

#include "image.h"
#include "kernel.h"
#include "./parallel/convolution.h"
#include "./sequential/convolution.h"


int main() {
    Image image("images/4K.jpg", 0, false);
    Kernel kernel = Kernel::unsharpen_mask_kernel();

    std::cout << kernel << std::endl;

    /*Image s = Sequential::Convolution::convolve(image, kernel, PaddingType::MIRROR);

    s.save_image("images/sequential.png");*/
    
    
    Image a = Parallel::Convolution::convolve_global(image, kernel, PaddingType::MIRROR);

    //a.save_image("images/global.png");

    
    Image b = Parallel::Convolution::convolve_constant(image, kernel, PaddingType::MIRROR);
   
    //b.save_image("images/constant.png");
    
    
    Image c = Parallel::Convolution::convolve_shared(image, kernel, PaddingType::MIRROR);

    //c.save_image("images/shared.png");

    std::cout << (c == a) << std::endl;


    Image d = Parallel::Convolution::convolve_pinned(image, kernel, PaddingType::MIRROR);

    //d.save_image("images/pinned.png");

    return 0;
}
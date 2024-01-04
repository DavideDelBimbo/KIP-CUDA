#ifndef CONVOLUTION_SEQUENTIAL_H
#define CONVOLUTION_SEQUENTIAL_H

#include <cmath>

#include "../image.h"
#include "../kernel.h"


namespace Sequential {
    class Convolution {
        public:
            /*
                * Convolve the image and measure the execution time.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The padding type to be applied.
                * @param results_path The path to save the results.
                * 
                * @return The convolved image.
            */
            static Image convolve(const Image& image, const Kernel& kernel, PaddingType padding_type = PaddingType::ZERO, std::string results_path = "");


        private:
            /*
                * Applies convolution to the image.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padded_image The padded image.
                * 
                * @return The convolved image.
            
            */
            static Image convolution(const Image& image, const Kernel& kernel, const Image& padded_image);
    };
}

#endif // CONVOLUTION_SEQUENTIAL_H
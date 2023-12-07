#ifndef CONVOLUTION_SEQUENTIAL_H
#define CONVOLUTION_SEQUENTIAL_H

#include <cmath>

#include "../image.h"
#include "../kernel.h"


namespace Sequential {
    class Convolution {
        public:
            /*
                * Applies convolution to the image.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * 
                * @return The convolved image.
            */
            static Image convolve(const Image& image, const Kernel& kernel, PaddingType padding_type = ZERO);
    };
}

#endif // CONVOLUTION_SEQUENTIAL_H
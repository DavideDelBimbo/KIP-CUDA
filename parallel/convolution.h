#ifndef CONVOLUTION_PARALLEL_H
#define CONVOLUTION_PARALLEL_H

#include "../image.h"
#include "../kernel.h"


namespace Parallel {
    class Convolution {
        public:
            /*
                * Applies convolution to an image using a kernel in global memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * 
                * @return The convolved image.
            */
            static Image convolve_global(const Image& image, const Kernel& kernel, PaddingType padding_type = ZERO);
    
            /*
                * Applies convolution to an image using a kernel in constant memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * 
                * @return The convolved image.
            */
            static Image convolve_constant(const Image& image, const Kernel& kernel, PaddingType padding_type = ZERO);
    
            /*
                * Applies convolution to an image using a kernel in shared memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * 
                * @return The convolved image.
            */
            static Image convolve_shared(const Image& image, const Kernel& kernel, PaddingType padding_type = ZERO);
    };    
}

#endif // CONVOLUTION_PARALLEL_H
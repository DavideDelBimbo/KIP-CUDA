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
                * @param results_path The path to save the results.
                * 
                * @return The convolved image.
            */
            static Image convolve_global(const Image& image, const Kernel& kernel, const PaddingType padding_type = PaddingType::ZERO, const std::string results_path = "");
    
            /*
                * Applies convolution to an image using a kernel in constant memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * @param results_path The path to save the results.
                * 
                * @return The convolved image.
            */
            static Image convolve_constant(const Image& image, const Kernel& kernel, const PaddingType padding_type = PaddingType::ZERO, const std::string results_path = "");
    
            /*
                * Applies convolution to an image using a kernel in shared memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * @param results_path The path to save the results.
                * 
                * @return The convolved image.
            */
            static Image convolve_shared(const Image& image, const Kernel& kernel, const PaddingType padding_type = PaddingType::ZERO, const std::string results_path = "");

            /*
                * Applies convolution to an image using a kernel in shared memory and pinned memory.
                *
                * @param image The image to be convolved.
                * @param kernel The kernel to be applied.
                * @param padding_type The type of padding to be applied.
                * @param results_path The path to save the results.
                * @param stream_count The number of streams to be used.
                * 
                * @return The convolved image.
            */
            static Image convolve_pinned(const Image& image, const Kernel& kernel, const PaddingType padding_type = PaddingType::ZERO, const std::string results_path = "", const int stream_count = 1);
    };
}

#endif // CONVOLUTION_PARALLEL_H
#include "convolution.h"

#define clamp(start, x, end) std::min(std::max(start, x), end)


// Methods.

Image Sequential::Convolution::convolve(const Image& image, const Kernel& kernel, PaddingType padding_type) {
    // Get the input image dimensions.
    const int width = image.get_width(); // Image width.
    const int height = image.get_height(); // Image height.
    const int channels = image.get_channels(); // Image channels.

    // Get the kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.


    // Apply padding to the input image.
    const int padding_width = std::floor(kernel_width / 2); // Padding width.
    const int padding_height = std::floor(kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.


    // Initialize the output image.
    Image output_image(width, height, channels, image.get_is_SoA()); // Output image.


    // Iterate over the image.
    for (int channel = 0; channel < channels; channel++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Output value for the current pixel.
                float output_value = 0;

                // Iterate over the kernel.
                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        // Get the pixel index to be convolved.
                        const int col = x + kx - std::floor((float)kernel_width / 2) + padding_width; // Column index.
                        const int row = y + ky - std::floor((float)kernel_width / 2) + padding_height; // Row index.

                        // Convolve the pixel.
                        output_value += padded_image(col, row, channel) * kernel(kx, ky);
                    }
                }

                // Set the output value (clamped between 0 and 255).
                output_image(x, y, channel) = (uint8_t)clamp(0.0f, output_value, 255.0f);
            }
        }
    }

    return output_image;
}
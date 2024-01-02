#include <iostream>
#include <chrono>

#include "convolution.h"
#include "../params.h"

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
    const int padding_width = std::floor((float)kernel_width / 2); // Padding width.
    const int padding_height = std::floor((float)kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.


    // Initialize the output image data.
    Image output_image = Image(width, height, channels, image.get_is_SoA()); // Output image.


    // Print the execution information.
    if (VERBOSITY >= 1) std::cout << "Starting sequential convolution..." << std::endl;

    // Execution time.
    double execution_time = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        // Start iteration execution time.
        auto start_time = std::chrono::high_resolution_clock::now();
        if (VERBOSITY >= 2) std::cout << "\tIteration: " << i;

        // Convolve the image.
        output_image = convolution(image, kernel, padded_image);

        // End iteration execution time.
        auto end_time = std::chrono::high_resolution_clock::now();

        // Measure the iteration execution time.
        double iteration_execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        execution_time += iteration_execution_time;

        // Print the iteration execution time.
        if (VERBOSITY >= 2) std::cout << " - Executed in " << iteration_execution_time << " ms" << std::endl;
    }

    // Print the execution time.
    if (VERBOSITY >= 1) std::cout << "Sequential execution time: " << execution_time / ITERATIONS << " ms (average of " << ITERATIONS << " runs)" << std::endl;


    // Return the convolved image.
    return output_image;
}

Image Sequential::Convolution::convolution(const Image& image, const Kernel& kernel, const Image& padded_image) {
    // Get the input image dimensions.
    const int width = image.get_width(); // Image width.
    const int height = image.get_height(); // Image height.
    const int channels = image.get_channels(); // Image channels.

    // Get the kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.

    // Get the padded image dimensions.
    const int padded_width = padded_image.get_width(); // Padded image width.
    const int padded_height = padded_image.get_height(); // Padded image height.


    // Get padding dimensions.
    const int padding_width = (padded_width - width) / 2; // Padding width.
    const int padding_height = (padded_height - height) / 2; // Padding height.


    // Initialize the output image.
    Image output_image = Image(width, height, channels, image.get_is_SoA()); // Output image.


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

    // Return the convolved image.
    return output_image;
}

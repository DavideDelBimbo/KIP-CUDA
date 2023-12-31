#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#include "convolution.h"
#include "../params.h"
#include "../utils.h"

#define clamp(start, x, end) (fmin(fmax(start, x), end))
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)


// Check the return value of the CUDA runtime API call and exit the application if the call has failed.
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess){
        return;
    }
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}


// Constant memory to store the kernel.
__constant__ float c_kernel[MAX_MASK_WIDTH * MAX_MASK_WIDTH];


// Device methods.

// Get the pixel value at the specified coordinates.
__device__ uint8_t& get_pixel_value(uint8_t* d_input, const int col, const int row, const int channel, const int width, const int height, const int channels, const bool is_SoA) {
    // Get the 1D pixel index.
    const int pixel_index = is_SoA ? ((channel * width * height) + (row * width) + col) : ((row * width + col) * channels + channel);

    return d_input[pixel_index];
}

// Get the kernel value at the specified coordinates.
__device__ float& get_kernel_value(float* d_kernel, const int col, const int row, const int width, const int height) {
    // Get the 1D kernel index.
    const int kernel_index = (row * width) + col;

    return d_kernel[kernel_index];
}

// Get the pixel value at the specified coordinates from shared memory.
__device__ uint8_t& get_shared_pixel_value(uint8_t* s_data, const int col, const int row, const int width, const int height) {
    // Get the 1D pixel index.
    const int pixel_index = (row * width) + col;

    return s_data[pixel_index];
}

// Set the pixel value at the specified coordinates.
__device__ void set_pixel_value(uint8_t* d_input, const int col, const int row, const int channel, const int width, const int height, const int channels, const bool is_SoA, const uint8_t value) {
    // Get the 1D pixel index.
    const int pixel_index = is_SoA ? ((channel * width * height) + (row * width) + col) : ((row * width + col) * channels + channel);

    d_input[pixel_index] = (uint8_t)value;
}


// Kernels.

// Kernel function for convolution using global memory.
__global__ void convolution_kernel_global(uint8_t* d_input, float* d_kernel, uint8_t* d_output,
                                   int width, int height, int channels,
                                   int kernel_width, int kernel_height,
                                   int padding_width, int padding_height, bool is_SoA)
{
    // Calculate the global index in the output image.
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index.
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.


    // Check if the thread is within the image bounds.
    if(x < width && y < height) {
        for(int channel = 0; channel < channels; channel++) {
            // Output value for the current pixel.
            float output_value = 0.0f;

            // Iterate over the kernel.
            for(int ky = 0; ky < kernel_height; ky++) {
                for(int kx = 0; kx < kernel_width; kx++) {
                    // Get the pixel index to be convolved.
                    const int col = x + kx - floor((float)kernel_width / 2) + padding_width; // Column index.
                    const int row = y + ky - floor((float)kernel_height / 2) + padding_height; // Row index.

                    // Add the convolution value to the output value.
                    output_value += get_pixel_value(d_input, col, row, channel, padded_width, padded_height, channels, is_SoA) * get_kernel_value(d_kernel, kx, ky, kernel_width, kernel_height);
                }
            }

            // Store the output value in global memory.
            set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
        }
    }
}

// Kernel function for convolution using constant memory.
__global__ void convolution_kernel_constant(uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   int kernel_width, int kernel_height,
                                   int padding_width, int padding_height, bool is_SoA)
{
    // Calculate the global index in the output image.
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index.
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.


    // Check if the thread is within the image bounds.
    if(x < width && y < height) {
        for(int channel = 0; channel < channels; channel++) {
            // Output value for the current pixel.
            float output_value = 0.0f;

            // Iterate over the kernel.
            for(int ky = 0; ky < kernel_height; ky++) {
                for(int kx = 0; kx < kernel_width; kx++) {
                    // Get the pixel index to be convolved.
                    const int col = x + kx - floor((float)kernel_width / 2) + padding_width; // Column index.
                    const int row = y + ky - floor((float)kernel_height / 2) + padding_height; // Row index.

                    // Add the convolution value to the output value.
                    output_value += get_pixel_value(d_input, col, row, channel, padded_width, padded_height, channels, is_SoA) * get_kernel_value(c_kernel, kx, ky, kernel_width, kernel_height);
                }
            }

            // Store the output value in global memory.
            set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
        }
    }

}


// Kernel function for convolution using shared memory.
__global__ void convolution_kernel_shared(uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   int kernel_width, int kernel_height,
                                   int padding_width, int padding_height, bool is_SoA)
{
    // Shared memory for the input image tile (dynamically sized by kernel launcher).
	extern __shared__ uint8_t s_data[];

    // Shared memory dimensions.
    const int s_width = TILE_WIDTH + (kernel_width - 1); // Shared memory width.
    const int s_height = TILE_WIDTH + (kernel_height - 1); // Shared memory height.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.

    // Number of entire batches to load the input image tile into shared memory.
    const int num_entire_batches = floor((float)s_width * s_height / (TILE_WIDTH * TILE_WIDTH)); // Number of entire batches.

    for(int channel = 0; channel < channels; channel++) {
        for(int i = 0; i < num_entire_batches; i++) {
            /*Loading firsts (TILE_WIDTH * TILE_WIDTH) elements into shared memory.*/

            // Shared memory index.
            int s_index = threadIdx.y * TILE_WIDTH + threadIdx.x + i * (TILE_WIDTH * TILE_WIDTH); // Shared memory index.
            int s_x = s_index % s_width; // Shared memory column index.
            int s_y = s_index / s_width; // Shared memory row index.

            // Global memory index to load the input image tile.
            int x = blockIdx.x * TILE_WIDTH + s_x - floor((float) kernel_width / 2) + padding_width; // Global memory column index.
            int y = blockIdx.y * TILE_WIDTH + s_y - floor((float) kernel_height / 2) + padding_width; // Global memory row index.

            // Check if the thread is within the image bounds.
            if(x >= 0 && x < padded_width && y >= 0 && y < padded_height) {
                // Load the pixel value into shared memory.
                s_data[s_index] = get_pixel_value(d_input, x, y, channel, padded_width, padded_height, channels, is_SoA);
            } else {
                // Load 0 into shared memory.
                s_data[s_index] = 0;
            }
        }
    
    
        /*Loading last (s_width * s_height) - (TILE_WIDTH * TILE_WIDTH) elements into shared memory.*/

        // Shared memory index.
        int s_index = (threadIdx.y * TILE_WIDTH) + threadIdx.x + num_entire_batches * (TILE_WIDTH * TILE_WIDTH); // Shared memory index.
        int s_x = s_index % s_width; // Shared memory column index.
        int s_y = s_index / s_width; // Shared memory row index.

        // Global memory index to load the input image tile.
        int x = (blockIdx.x * TILE_WIDTH) + s_x - floor((float) kernel_width / 2) + padding_width; // Global memory column index.
        int y = (blockIdx.y * TILE_WIDTH) + s_y - floor((float) kernel_height / 2) + padding_height; // Global memory row index.

        // Check if the thread is within the image bounds.
        if(s_y < s_height) {
            if(x >= 0 && x < padded_width && y >= 0 && y < padded_height) {
                // Load the pixel value into shared memory.
                s_data[s_index] = get_pixel_value(d_input, x, y, channel, padded_width, padded_height, channels, is_SoA);
            } else {
                // Load 0 into shared memory.
                s_data[s_index] = 0;
            }
        }

        // Wait for all threads to finish loading the input image tile into shared memory.
        __syncthreads();


        /*Convolve the input image tile with the kernel.*/

        // Output value for the current pixel.
        float output_value = 0.0f;

        // Iterate over the kernel.
        for(int ky = 0; ky < kernel_height; ky++) {
            for(int kx = 0; kx < kernel_width; kx++) {
                // Get the pixel index to be convolved.
                const int col = threadIdx.x + kx; // Column index.
                const int row = threadIdx.y + ky; // Row index.

                // Add the convolution value to the output value.
                output_value += get_shared_pixel_value(s_data, col, row, s_width, s_height) * get_kernel_value(c_kernel, kx, ky, kernel_width, kernel_height);
            }
        }


        // Calculate the global index in the output image.
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;

        // Store the output value in global memory.
        if (x < width && y < height) {
            set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
        }

        // Wait for all threads to finish convolving the input image tile with the kernel.
        __syncthreads();
    }
}


// Methods.

Image Parallel::Convolution::convolve_global(const Image& image, const Kernel& kernel, const PaddingType padding_type, const std::string results_path) {
    // Input image dimensions.
    const int width = image.get_width(); // Input image width.
    const int height = image.get_height(); // Input image height.
    const int channels = image.get_channels(); // Input image channels.

    // Kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.


    // Apply padding to the input image.
    const int padding_width = floor((float)kernel_width / 2); // Padding width.
    const int padding_height = floor((float)kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.

    // Padded image dimensions.
    const int padded_width = padded_image.get_width(); // Padded image width.
    const int padded_height = padded_image.get_height(); // Padded image height.
    const int padded_channels = padded_image.get_channels(); // Padded image channels.


    // Sizes in bytes.
    const int input_size = padded_width * padded_height * padded_channels * sizeof(uint8_t); // Input image size.
    const int output_size = width * height * channels * sizeof(uint8_t); // Output image size.
    const int kernel_size = kernel_width * kernel_height * sizeof(float); // Kernel size.
    

    // Host memory pointers.
    uint8_t* h_input = padded_image.get_data(); // Input image data.
    uint8_t* h_output = (uint8_t*)malloc(output_size); // Output image data.
    float* h_kernel = kernel.get_data(); // Kernel data.

    // Device memory pointers.
    uint8_t* d_input; // Input image data.
    uint8_t* d_output; // Output image data.
    float* d_kernel; // Kernel data.


    // Allocate device memory.
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, output_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_kernel, kernel_size));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Threads per block.
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Blocks per grid.


    // Execution time.
    float execution_time = 0;

    // Print the execution information.
    if (VERBOSITY >= 1) std::cout << "Starting parallel convolution with global memory..." << std::endl;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // CUDA events for measuring execution time.
        cudaEvent_t start, stop;
        CUDA_CHECK_RETURN(cudaEventCreate(&start));
        CUDA_CHECK_RETURN(cudaEventCreate(&stop));


        // Copy data from host to device global memory.
        CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));


        // Start iteration execution time.
        if (VERBOSITY >= 2) std::cout << "\tIteration: " << i;
        CUDA_CHECK_RETURN(cudaEventRecord(start));

        // Launch kernel.
        convolution_kernel_global<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());
        
        // End iteration execution time.
        CUDA_CHECK_RETURN(cudaEventRecord(stop));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        // Copy output data from device global memory to host memory.
        CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop)); 

        // Measure the iteration execution time.
        float iteration_execution_time = 0;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&iteration_execution_time, start, stop));
        execution_time += iteration_execution_time;


        // Destroy the CUDA events.
        CUDA_CHECK_RETURN(cudaEventDestroy(start));
        CUDA_CHECK_RETURN(cudaEventDestroy(stop));


        // Print the iteration execution time.
        if (VERBOSITY >= 2) std::cout << " - Execution: " << iteration_execution_time << " ms" << std::endl;
    }

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
    CUDA_CHECK_RETURN(cudaFree(d_kernel));


    // Print the execution time.
    if (VERBOSITY >= 1) std::cout << "Execution time with global memory: " << execution_time / ITERATIONS << " ms (average of " << ITERATIONS << " runs)" << std::endl;

    // Save the results.
    if (!results_path.empty()) {
        std::string execution_type = "global";
        save_results(results_path, execution_type, width, height, channels, image.get_is_SoA(), kernel_width, kernel_height, execution_time / ITERATIONS, ITERATIONS);
    }

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

Image Parallel::Convolution::convolve_constant(const Image &image, const Kernel &kernel, const PaddingType padding_type, const std::string results_path) {
    // Input image dimensions.
    const int width = image.get_width(); // Input image width.
    const int height = image.get_height(); // Input image height.
    const int channels = image.get_channels(); // Input image channels.

    // Kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.


    // Apply padding to the input image.
    const int padding_width = floor((float)kernel_width / 2); // Padding width.
    const int padding_height = floor((float)kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.

    // Padded image dimensions.
    const int padded_width = padded_image.get_width(); // Padded image width.
    const int padded_height = padded_image.get_height(); // Padded image height.
    const int padded_channels = padded_image.get_channels(); // Padded image channels.


    // Sizes in bytes.
    const int input_size = padded_width * padded_height * padded_channels * sizeof(uint8_t); // Input image size.
    const int output_size = width * height * channels * sizeof(uint8_t); // Output image size.
    const int kernel_size = kernel_width * kernel_height * sizeof(float); // Kernel size.
    

    // Host memory pointers.
    uint8_t* h_input = padded_image.get_data(); // Input image data.
    uint8_t* h_output = (uint8_t*)malloc(output_size); // Output image data.
    float* h_kernel = kernel.get_data(); // Kernel data.

    // Device memory pointers.
    uint8_t* d_input; // Input image data.
    uint8_t* d_output; // Output image data.


    // Allocate device memory.
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, output_size));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Threads per block.
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Blocks per grid.


    // Execution time.
    float execution_time = 0;

    // Print the execution information.
    if (VERBOSITY >= 1) std::cout << "Starting parallel convolution with constant memory..." << std::endl;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // CUDA events for measuring execution time.
        cudaEvent_t start, stop;
        CUDA_CHECK_RETURN(cudaEventCreate(&start));
        CUDA_CHECK_RETURN(cudaEventCreate(&stop));


        // Copy input data from host to device global memory.
        CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

        // Copy kernel data from host to device constant memory.
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size, 0, cudaMemcpyHostToDevice));


        // Start iteration execution time.
        if (VERBOSITY >= 2) std::cout << "\tIteration: " << i;
        CUDA_CHECK_RETURN(cudaEventRecord(start));

        // Launch kernel.
        convolution_kernel_constant<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());
        
        // End iteration execution time.
        CUDA_CHECK_RETURN(cudaEventRecord(stop));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        

        // Copy output data from device global memory to host memory.
        CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop)); 

        // Measure the iteration execution time.
        float iteration_execution_time = 0;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&iteration_execution_time, start, stop));
        execution_time += iteration_execution_time;


        // Destroy the CUDA events.
        CUDA_CHECK_RETURN(cudaEventDestroy(start));
        CUDA_CHECK_RETURN(cudaEventDestroy(stop));


        // Print the iteration execution time.
        if (VERBOSITY >= 2) std::cout << " - Execution: " << iteration_execution_time << " ms" << std::endl;
    }

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));


    // Print the execution time.
    if (VERBOSITY >= 1) std::cout << "Execution time with constant memory: " << execution_time / ITERATIONS << " ms (average of " << ITERATIONS << " runs)" << std::endl;

    // Save the results.
    if (!results_path.empty()) {
        std::string execution_type = "constant";
        save_results(results_path, execution_type, width, height, channels, image.get_is_SoA(), kernel_width, kernel_height, execution_time / ITERATIONS, ITERATIONS);
    }

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

Image Parallel::Convolution::convolve_shared(const Image &image, const Kernel &kernel, const PaddingType padding_type, const std::string results_path) {
    // Input image dimensions.
    const int width = image.get_width(); // Input image width.
    const int height = image.get_height(); // Input image height.
    const int channels = image.get_channels(); // Input image channels.

    // Kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.


    // Apply padding to the input image.
    const int padding_width = floor((float)kernel_width / 2); // Padding width.
    const int padding_height = floor((float)kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.

    // Padded image dimensions.
    const int padded_width = padded_image.get_width(); // Padded image width.
    const int padded_height = padded_image.get_height(); // Padded image height.
    const int padded_channels = padded_image.get_channels(); // Padded image channels.


    // Sizes in bytes.
    const int input_size = padded_width * padded_height * padded_channels * sizeof(uint8_t); // Input image size.
    const int output_size = width * height * channels * sizeof(uint8_t); // Output image size.
    const int kernel_size = kernel_width * kernel_height * sizeof(float); // Kernel size.
    const int shared_size = (TILE_WIDTH + kernel_width - 1) * (TILE_WIDTH + kernel_height - 1) * sizeof(uint8_t); // Shared memory size.
    

    // Host memory pointers.
    uint8_t* h_input = padded_image.get_data(); // Input image data.
    uint8_t* h_output = (uint8_t*)malloc(output_size); // Output image data.
    float* h_kernel = kernel.get_data(); // Kernel data.

    // Device memory pointers.
    uint8_t* d_input; // Input image data.
    uint8_t* d_output; // Output image data.


    // Allocate device memory.
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, output_size));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Threads per block.
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Blocks per grid.


    // Execution time.
    float execution_time = 0;

    // Print the execution information.
    if (VERBOSITY >= 1) std::cout << "Starting parallel convolution with shared memory..." << std::endl;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // CUDA events for measuring execution time.
        cudaEvent_t start, stop;
        CUDA_CHECK_RETURN(cudaEventCreate(&start));
        CUDA_CHECK_RETURN(cudaEventCreate(&stop));


        // Copy input data from host to device global memory.
        CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

        // Copy kernel data from host to device constant memory.
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size, 0, cudaMemcpyHostToDevice));


        // Start iteration execution time.
        if (VERBOSITY >= 2) std::cout << "\tIteration: " << i;
        CUDA_CHECK_RETURN(cudaEventRecord(start));

        // Launch kernel.
        convolution_kernel_shared<<<gridDim, blockDim, shared_size>>>(d_input, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());

        // End iteration execution time.
        CUDA_CHECK_RETURN(cudaEventRecord(stop));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        // Copy output data from device global memory to host memory.
        CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

        // Measure the iteration execution time.
        float iteration_execution_time = 0;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&iteration_execution_time, start, stop));
        execution_time += iteration_execution_time;


        // Destroy the CUDA events.
        CUDA_CHECK_RETURN(cudaEventDestroy(start));
        CUDA_CHECK_RETURN(cudaEventDestroy(stop));


        // Print the iteration execution time.
        if (VERBOSITY >= 2) std::cout << " - Execution: " << iteration_execution_time << " ms" << std::endl;
    }

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));


    // Print the execution time.
    if (VERBOSITY >= 1) std::cout << "Execution time with shared memory: " << execution_time / ITERATIONS << " ms (average of " << ITERATIONS << " runs)" << std::endl;

    // Save the results.
    if (!results_path.empty()) {
        std::string execution_type = "shared";
        save_results(results_path, execution_type, width, height, channels, image.get_is_SoA(), kernel_width, kernel_height, execution_time / ITERATIONS, ITERATIONS);
    }

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

Image Parallel::Convolution::convolve_pinned(const Image &image, const Kernel &kernel, const PaddingType padding_type, const std::string results_path, const int stream_count) {
    // Input image dimensions.
    const int width = image.get_width(); // Input image width.
    const int height = image.get_height(); // Input image height.
    const int channels = image.get_channels(); // Input image channels.

    // Kernel dimensions.
    const int kernel_width = kernel.get_width(); // Kernel width.
    const int kernel_height = kernel.get_height(); // Kernel height.


    // Apply padding to the input image.
    const int padding_width = floor((float)kernel_width / 2); // Padding width.
    const int padding_height = floor((float)kernel_height / 2); // Padding height.
    Image padded_image = image.padding(padding_width, padding_height, padding_type); // Padded image.

    // Padded image dimensions.
    const int padded_width = padded_image.get_width(); // Padded image width.
    const int padded_height = padded_image.get_height(); // Padded image height.
    const int padded_channels = padded_image.get_channels(); // Padded image channels.


    // Sizes in bytes.
    const int input_size = padded_width * padded_height * padded_channels * sizeof(uint8_t); // Input image size.
    const int output_size = width * height * channels * sizeof(uint8_t); // Output image size.
    const int kernel_size = kernel_width * kernel_height * sizeof(float); // Kernel size.
    const int shared_size = (TILE_WIDTH + kernel_width - 1) * (TILE_WIDTH + kernel_height - 1) * sizeof(uint8_t); // Shared memory size.


    // Pageable host memory pointers.
    uint8_t* h_input = padded_image.get_data(); // Input image data.
    uint8_t* h_output = (uint8_t*)malloc(output_size); // Output image data.
    float* h_kernel = kernel.get_data(); // Kernel data.

    // Pinned host memory pointers.
    uint8_t* h_pinned_input; // Input image data.
    uint8_t* h_pinned_output; // Output image data.

    // Device memory pointers.
    uint8_t* d_input; // Input image data.
    uint8_t* d_output; // Output image data.


    // Allocate pinned host memory.
    CUDA_CHECK_RETURN(cudaMallocHost((void**)&h_pinned_input, input_size));
    CUDA_CHECK_RETURN(cudaMallocHost((void**)&h_pinned_output, output_size));

    // Allocate device memory.
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, input_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, output_size));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Threads per block.
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Blocks per grid.


    // Execution time.
    float execution_time = 0;

    // Print the execution information.
    if (VERBOSITY >= 1) std::cout << "Starting parallel convolution with pinned memory..." << std::endl;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // CUDA events for measuring execution time.
        cudaEvent_t start, stop;
        CUDA_CHECK_RETURN(cudaEventCreate(&start));
        CUDA_CHECK_RETURN(cudaEventCreate(&stop));


        // Create CUDA streams for asynchronous memory transfers.
        cudaStream_t* streams = new cudaStream_t[stream_count];
        for (int n = 0; n < stream_count; n++) {
            CUDA_CHECK_RETURN(cudaStreamCreate(&streams[n]));
        }


        // Copy input data from pageable host memory to pinned host memory.
        CUDA_CHECK_RETURN(cudaMemcpy(h_pinned_input, h_input, input_size, cudaMemcpyHostToHost));

        // Copy kernel data from host to device constant memory.
        CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(c_kernel, h_kernel, kernel_size, 0, cudaMemcpyHostToDevice));

        
        // Start iteration execution time.
        if (VERBOSITY >= 2) std::cout << "\tIteration: " << i;
        CUDA_CHECK_RETURN(cudaEventRecord(start));

        for (int n = 0; n < stream_count; n++)  {
            // Calculate the stream sizes.
            int input_stream_size = input_size / stream_count; // Input stream size.
            int input_offset = n * input_stream_size; // Offset for the input stream.

            // Copy input data from pinned host memory to device global memory asynchronously.
            CUDA_CHECK_RETURN(cudaMemcpyAsync(d_input + input_offset, h_pinned_input + input_offset, input_stream_size, cudaMemcpyHostToDevice, streams[n]));

            // Launch kernel.
            convolution_kernel_shared<<<gridDim, blockDim, shared_size, streams[n]>>>(d_input + input_offset, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());

            // Copy output data from device global memory to pinned host memory asynchronously.
            CUDA_CHECK_RETURN(cudaMemcpyAsync(h_pinned_output, d_output, output_size, cudaMemcpyDeviceToHost, streams[n]));
        }
        
        // End iteration execution time.
        CUDA_CHECK_RETURN(cudaEventRecord(stop));


        // Waits for streams to finish work.
        CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[0]));


        // Copy output data from pinned host memory to pageable host memory.
        CUDA_CHECK_RETURN(cudaMemcpy(h_output, h_pinned_output, output_size, cudaMemcpyHostToHost));


        // Wait for the kernel to finish execution.
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

        // Measure the iteration execution time.
        float iteration_execution_time = 0;
        CUDA_CHECK_RETURN(cudaEventElapsedTime(&iteration_execution_time, start, stop));
        execution_time += iteration_execution_time;


        // Destroy the CUDA streams.
        for (int n = 0; n < stream_count; n++) {
            CUDA_CHECK_RETURN(cudaStreamDestroy(streams[n]));
        }
        
        // Deallocate the array of streams.
        delete[] streams;

        // Destroy the CUDA events.
        CUDA_CHECK_RETURN(cudaEventDestroy(start));
        CUDA_CHECK_RETURN(cudaEventDestroy(stop));


        // Print the iteration execution time.
        if (VERBOSITY >= 2) std::cout << " - Execution: " << iteration_execution_time << " ms" << std::endl;
    }

    // Clean up pinned host memory.
    CUDA_CHECK_RETURN(cudaFreeHost(h_pinned_input));
    CUDA_CHECK_RETURN(cudaFreeHost(h_pinned_output));

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));


    // Print the execution time.
    if (VERBOSITY >= 1) std::cout << "Execution time with pinned memory: " << execution_time / ITERATIONS << " ms (average of " << ITERATIONS << " runs)" << std::endl;

    // Save the results.
    if (!results_path.empty()) {
        std::string execution_type = "pinned";
        save_results(results_path, execution_type, width, height, channels, image.get_is_SoA(), kernel_width, kernel_height, execution_time / ITERATIONS, ITERATIONS);
    }
    
    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

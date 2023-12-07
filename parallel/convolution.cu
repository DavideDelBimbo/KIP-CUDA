#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "convolution.h"

#define TILE_WIDTH 32
#define MAX_MASK_WIDTH 10

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


// Get the pixel value at the specified coordinates.
__device__ uint8_t& get_pixel_value(uint8_t* d_input, const int col, const int row, const int channel, const int width, const int height, const int channels, const bool is_SoA) {
    // Get the 1D pixel index.
    const int pixel_index = is_SoA ? ((channel * width * height) + (row * width) + col) : ((row * width + col) * channels + channel);

    return d_input[pixel_index];
}

// Set the pixel value at the specified coordinates.
__device__ void set_pixel_value(uint8_t* d_input, const int col, const int row, const int channel, const int width, const int height, const int channels, const bool is_SoA, const uint8_t value) {
    // Get the 1D pixel index.
    const int pixel_index = is_SoA ? ((channel * width * height) + (row * width) + col) : ((row * width + col) * channels + channel);

    d_input[pixel_index] = (uint8_t)value;
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


// Kernel function for convolution using global memory.
__global__ void convolution_kernel_global(uint8_t* d_input, float* d_kernel, uint8_t* d_output,
                                   int width, int height, int channels,
                                   int kernel_width, int kernel_height,
                                   int padding_width, int padding_height, bool is_SoA)
{
    // Calculate the global index in the output image.
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index.
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index.
    const int channel = blockIdx.z * blockDim.z + threadIdx.z; // Channel index.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.


    // Check if the thread is within the image bounds.
    if(x < width && y < height && channel < channels) {
        // Output value for the current pixel.
        float output_value = 0.0f;

        // Iterate over the kernel.
        for(int ky = 0; ky < kernel_height; ky++) {
            for(int kx = 0; kx < kernel_width; kx++) {
                // Get the pixel index to be convolved.
                const int col = x + kx - floor((float)kernel_width / 2) + padding_width;
                const int row = y + ky - floor((float)kernel_height / 2) + padding_height; 

                // Add the convolution value to the output value.
                output_value += get_pixel_value(d_input, col, row, channel, padded_width, padded_height, channels, is_SoA) * get_kernel_value(d_kernel, kx, ky, kernel_width, kernel_height);
            }
        }

        // Store the output value in global memory.
        set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
    }
}


// Constant memory to store the kernel.
__constant__ float c_kernel[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

// Kernel function for convolution using constant memory.
__global__ void convolution_kernel_constant(uint8_t* d_input, uint8_t* d_output,
                                   int width, int height, int channels,
                                   int kernel_width, int kernel_height,
                                   int padding_width, int padding_height, bool is_SoA)
{
    // Calculate the global index in the output image.
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index.
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index.
    const int channel = blockIdx.z * blockDim.z + threadIdx.z; // Channel index.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.


    // Check if the thread is within the image bounds.
    if(x < width && y < height && channel < channels) {
        // Output value for the current pixel.
        float output_value = 0.0f;

        // Iterate over the kernel.
        for(int ky = 0; ky < kernel_height; ky++) {
            for(int kx = 0; kx < kernel_width; kx++) {
                // Get the pixel index to be convolved.
                const int col = x + kx - floor((float)kernel_width / 2) + padding_width;
                const int row = y + ky - floor((float)kernel_height / 2) + padding_height;

                // Add the convolution value to the output value.
                output_value += get_pixel_value(d_input, col, row, channel, padded_width, padded_height, channels, is_SoA) * get_kernel_value(c_kernel, kx, ky, kernel_width, kernel_height);
            }
        }

        // Store the output value in global memory.
        set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
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
    const int s_width = blockDim.x + (kernel_width - 1); // Shared memory width.
    const int s_height = blockDim.y + (kernel_height - 1); // Shared memory height.

    // Padded image dimensions.
    const int padded_width = width + (2 * padding_width); // Padded image width.
    const int padded_height = height + (2 * padding_height); // Padded image height.


    // Loading first (blockDim.x * blockDim.y) elements into shared memory.

    // Shared memory index.
    int s_index = (threadIdx.y * blockDim.x) + threadIdx.x; // Shared memory index.
    int s_x = s_index % s_width; // Shared memory column index.
    int s_y = s_index / s_width; // Shared memory row index.

    // Global memory index to load the input image tile.
    int x = (blockIdx.x * blockDim.x) + s_x - floor((float) kernel_width / 2); // Global memory column index.
    int y = (blockIdx.y * blockDim.y) + s_y - floor((float) kernel_height / 2); // Global memory row index.
    int channel = blockIdx.z * blockDim.z + threadIdx.z; // Global memory channel index.

    // Check if the thread is within the image bounds.
    if(x >= 0 && x < padded_width && y >= 0 && y < padded_height) {
        // Load the pixel value into shared memory.
        s_data[s_index] = get_pixel_value(d_input, x, y, channel, padded_width, padded_height, channels, is_SoA);
    } else {
        // Load 0 into shared memory.
        s_data[s_index] = 0;
    }
    
    
    // Loading last (s_width * s_height) - (blockDim.x * blockDim.y) elements into shared memory.

    // Shared memory index.
    s_index = (threadIdx.y * blockDim.x) + threadIdx.x + (blockDim.x * blockDim.y); // Shared memory index.
    s_x = s_index % s_width; // Shared memory column index.
    s_y = s_index / s_width; // Shared memory row index.

    // Global memory index to load the input image tile.
    x = (blockIdx.x * blockDim.x) + s_x - floor((float) kernel_width / 2); // Global memory column index.
    y = (blockIdx.y * blockDim.y) + s_y - floor((float) kernel_height / 2); // Global memory row index.
    channel = blockIdx.z * blockDim.z + threadIdx.z; // Global memory channel index.

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

    
    // Convolve the input image tile with the kernel.

    // Output value for the current pixel.
    float output_value = 0.0f;

    // Iterate over the kernel.
    for(int ky = 0; ky < kernel_height; ky++) {
        for(int kx = 0; kx < kernel_width; kx++) {
            // Get the pixel index to be convolved.
            const int col = threadIdx.x + kx;
            const int row = threadIdx.y + ky;

            // Add the convolution value to the output value.
            output_value += get_shared_pixel_value(s_data, col, row, s_width, s_height) * get_kernel_value(c_kernel, kx, ky, kernel_width, kernel_height);
        }
    }


    // Calculate the global index in the output image.
    x = blockIdx.x * blockDim.x + threadIdx.x - padding_width;
    y = blockIdx.y * blockDim.y + threadIdx.y - padding_height;
    channel = blockIdx.z * blockDim.z + threadIdx.z;

    // Store the output value in global memory.
    if (x >= 0 && x < width && y >= 0 && y < height && channel < channels) {
        set_pixel_value(d_output, x, y, channel, width, height, channels, is_SoA, (uint8_t)clamp(0.0f, output_value, 255.0f));
    }

    // Wait for all threads to finish convolving the input image tile with the kernel.
    //__syncthreads();
}


// Methods.

Image Parallel::Convolution::convolve_global(const Image& image, const Kernel& kernel, PaddingType padding_type) {
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

    // Copy data from host to device global memory.
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // Threads per block.
    dim3 gridDim(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH), channels); // Blocks per grid.

    // Launch kernel.
    convolution_kernel_global<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());

    // Waits for threads to finish work.
	cudaDeviceSynchronize();

    // Copy output data from device global memory to host memory.
    CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
    CUDA_CHECK_RETURN(cudaFree(d_kernel));

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

Image Parallel::Convolution::convolve_constant(const Image &image, const Kernel &kernel, PaddingType padding_type) {
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

    // Copy input data from host to device global memory.
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    // Copy kernel data from host to device constant memory.
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size, 0, cudaMemcpyHostToDevice));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // Threads per block.
    dim3 gridDim(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH), channels); // Blocks per grid.

    // Launch kernel.
    convolution_kernel_constant<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());

    // Waits for threads to finish work.
    cudaDeviceSynchronize();

    // Copy output data from device global memory to host memory.
    CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

Image Parallel::Convolution::convolve_shared(const Image &image, const Kernel &kernel, PaddingType padding_type) {
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

    // Copy input data from host to device global memory.
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    // Copy kernel data from host to device constant memory.
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size, 0, cudaMemcpyHostToDevice));


    // Specify block and grid dimensions.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // Threads per block.
    dim3 gridDim(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH), channels); // Blocks per grid.

    // Launch kernel.
    convolution_kernel_shared<<<gridDim, blockDim, shared_size>>>(d_input, d_output, width, height, channels, kernel_width, kernel_height, padding_width, padding_height, image.get_is_SoA());

    // Waits for threads to finish work.
    cudaDeviceSynchronize();
    
    // Copy output data from device global memory to host memory.
    CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Clean up device memory after kernel execution.
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));

    // Create the output image.
    return Image(width, height, channels, h_output, image.get_is_SoA());
}

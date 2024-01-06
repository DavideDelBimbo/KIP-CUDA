#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>
#include <vector>

#include "params.h"
#include "image.h"
#include "kernel.h"
#include "./parallel/convolution.h"
#include "./sequential/convolution.h"


std::string IMAGE_PATH = "";
static bool SOA = false;
static PaddingType PADDING_TYPE = PaddingType::MIRROR;
static std::string KERNEL = "";
static int KERNEL_SIZE = 0;
static float* KERNEL_DATA = nullptr;
static bool KERNEL_NORMALIZATION = false;
static std::string EXECUTION_TYPE = "";
static std::string MEMORY_TYPE = "";
static std::string OUTPUT_PATH = "";
static std::string RESULTS_PATH = ".\\results\\";

void printHelp() {
    std::cout << "Kernel Image Processing CUDA Help:" << std::endl;
    std::cout << "Usage: ./kip [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h: Display this help message." << std::endl;
    std::cout << "  --image_path, -I: Path to the input image file." << std::endl;
    std::cout << "  --SoA, -S: Use Structure of Arrays (SoA) data layout." << std::endl;
    std::cout << "  --padding_type, -P: Padding type ('zero', 'replicate' or 'mirror')." << std::endl;
    std::cout << "  --kernel, -K: Kernel type ('box_blur', 'gaussian_blur', 'sharpen', 'edge_detection', 'unsharpen_mask', 'emboss' or 'custom')." << std::endl;
    std::cout << "  --kernel_size, -Z: Size of the custom kernel (required 'custom' kernel)." << std::endl;
    std::cout << "  --kernel_data, -D: Data of the custom kernel (required 'custom' kernel with specific 'kernel_size')." << std::endl;
    std::cout << "  --kernel_normalization, -N: Normalization of the custom kernel (required 'custom' kernel)." << std::endl;
    std::cout << "  --execution_type, -E: Execution type ('parallel' or 'sequential')." << std::endl;
    std::cout << "  --memory_type, -M: Memory management type ('global', 'constant', 'shared' or 'pinned')." << std::endl;
    std::cout << "  --output_path, -O: Path to the output image file." << std::endl;
    std::cout << "  --results_path, -R: Base path for the results (default: './results/')." << std::endl;
}

int processInput(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        // Get the argument.
        const char *arg = argv[i];

        // Check if the argument is a flag.
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-H") == 0) {
            // Print help and exit.
            printHelp();
            return 0;
        } else if (strncmp(arg, "--image_path=", 12) == 0 || strncmp(arg, "-I=", 3) == 0) {
            IMAGE_PATH = strchr(arg, '=') + 1;
        } else if (strcmp(arg, "--SoA") == 0 || strcmp(arg, "-S") == 0) {
            SOA = true;
        } else if (strncmp(arg, "--padding_type=", 15) == 0 || strncmp(arg, "-P=", 3) == 0) {
            // Set the padding type.
            const char *value = strchr(arg, '=') + 1;

            if (strcmp(value, "zero") == 0) {
                // Zero padding.
                PADDING_TYPE = PaddingType::ZERO;
            } else if (strcmp(value, "replicate") == 0) {
                // Replicate padding.
                PADDING_TYPE = PaddingType::REPLICATE;
            } else if (strcmp(value, "mirror") == 0) {
                // Mirror padding.
                PADDING_TYPE = PaddingType::MIRROR;
            } else {
                // Invalid padding type.
                std::cerr << "Invalid argument for padding type." << std::endl;
                return 1;
            }
        } else if (strncmp(arg, "--kernel=", 9) == 0 || strncmp(arg, "-K=", 3) == 0) {
            // Set the kernel.
            const char *value = strchr(arg, '=') + 1;

            if (strcmp(value, "box_blur") == 0) {
                // Box blur kernel.
                KERNEL = "box_blur";
            } else if (strcmp(value, "gaussian_blur") == 0) {
                // Gaussian blur kernel.
                KERNEL = "gaussian_blur";
            } else if (strcmp(value, "sharpen") == 0) {
                // Sharpen kernel.
                KERNEL = "sharpen";
            } else if (strcmp(value, "edge_detection") == 0) {
                // Edge detection kernel.
                KERNEL = "edge_detection";
            } else if (strcmp(value, "unsharpen_mask") == 0) {
                // Unsharpen mask kernel.
                KERNEL = "unsharpen_mask";
            } else if (strcmp(value, "emboss") == 0) {
                // Emboss kernel.
                KERNEL = "emboss";
            } else if (strcmp(value, "custom") == 0) {
                // Custom kernel.
                KERNEL = "custom";
            } else {
                // Invalid kernel.
                std::cerr << "Invalid argument for kernel." << std::endl;
                return 1;
            }
        } else if ((KERNEL == "custom") && (strncmp(arg, "--kernel_size=", 14) == 0 || strncmp(arg, "-Z=", 3) == 0)) {
            KERNEL_SIZE = std::stoi(strchr(arg, '=') + 1);

            if(KERNEL_SIZE <= 0) {
                // Invalid kernel size.
                std::cerr << "Invalid argument for kernel size." << std::endl;
                return 1;
            }
        } else if ((KERNEL == "custom") && (strncmp(arg, "--kernel_data=", 14) == 0 || strncmp(arg, "-D=", 3) == 0)) {
            // Set the kernel data.
            const char *value = strchr(arg, '=') + 1;

            // Initialize the kernel data.
            KERNEL_DATA = new float[KERNEL_SIZE * KERNEL_SIZE];

            // Split the kernel data.
            std::stringstream ss(value); // Input string stream.
            float num; // Input number.
            size_t i = 0; // Index of the kernel data.
            while (ss >> num) {
                // Check if the number is followed by an operation.
                char op;
                if (ss.peek() == '+' || ss.peek() == '-' || ss.peek() == '*' || ss.peek() == '/') {
                    ss >> op; // Get the operation.
                    float nextNum;
                    ss >> nextNum; // Get the next number.

                    // Perform the operation.
                    if (op == '+') num += nextNum;
                    else if (op == '-') num -= nextNum;
                    else if (op == '*') num *= nextNum;
                    else if (op == '/') num /= nextNum;
                }
                // Save the number.
                KERNEL_DATA[i++] = num;

                if (ss.peek() == ',') {
                    // Ignore the comma.
                    ss.ignore();
                }
            }

            if (i != KERNEL_SIZE * KERNEL_SIZE) {
                // Invalid kernel data.
                std::cerr << "Invalid argument for kernel data." << std::endl;
                return 1;
            }
        } else if ((KERNEL == "custom") && (strcmp(arg, "--kernel_normalization") == 0 || strcmp(arg, "-N") == 0)) {
            // Normalize the kernel.
            KERNEL_NORMALIZATION = true;
        } else if (strncmp(arg, "--execution_type=", 17) == 0 || strncmp(arg, "-E=", 3) == 0) {
            // Set the execution type.
            const char *value = strchr(arg, '=') + 1;

            if (strcmp(value, "parallel") == 0) {
                // Parallel execution.
                EXECUTION_TYPE = "parallel";
            } else if (strcmp(value, "sequential") == 0) {
                // Sequential execution.
                EXECUTION_TYPE = "sequential";
            } else {
                // Invalid execution type.
                std::cerr << "Invalid argument for execution type." << std::endl;
                return 1;
            }
        } else if ((EXECUTION_TYPE == "parallel") && (strncmp(arg, "--memory_type=", 14) == 0 || strncmp(arg, "-M=", 3) == 0)) {
            // Set the memory type.
            const char *value = strchr(arg, '=') + 1;

            if (strcmp(value, "global") == 0) {
                // Global memory.
                MEMORY_TYPE = "global";
            } else if (strcmp(value, "constant") == 0) {
                // Constant memory.
                MEMORY_TYPE = "constant";
            } else if (strcmp(value, "shared") == 0) {
                // Shared memory.
                MEMORY_TYPE = "shared";
            } else if (strcmp(value, "pinned") == 0) {
                // Pinned memory.
                MEMORY_TYPE = "pinned";
            } else {
                // Invalid memory type.
                std::cerr << "Invalid argument for memory type." << std::endl;
                return 1;
            }
        } else if (strncmp(arg, "--output_path=", 14) == 0 || strncmp(arg, "-O=", 3) == 0) {
            OUTPUT_PATH = strchr(arg, '=') + 1;
        } else if (strncmp(arg, "--results_path=", 12) == 0 || strncmp(arg, "-R=", 3) == 0) {
             if(arg[strlen(arg)-1] == '\\') {
                // Set the base path for the results.
                RESULTS_PATH = strchr(arg, '=') + 1;
            } else {
                // Invalid base path.
                std::cerr << "Invalid argument for base path. Please specify a valid base path." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Invalid argument: " << arg << ". Use '--help' or '-h' for usage instructions." << std::endl;
            return 1;
        }
    }

    if (IMAGE_PATH == "" || KERNEL == "" || EXECUTION_TYPE == "") {
        std::cout << "Please specify valid values for required parameters." << std::endl;
        return 1;
    }

    return 0;
}

// Run the convolution on the image with the kernel and save the result.
void runConvolution(const Image& image, const Kernel& kernel) {
    // Print the kernel.
    if (VERBOSITY >= 1) std::cout << kernel << std::endl;

    // Run the convolution.
    if (EXECUTION_TYPE == "sequential") {
        // Run the sequential convolution.
        Image result = Sequential::Convolution::convolve(image, kernel, PADDING_TYPE, RESULTS_PATH);

        // Save the convolved image.
        if (!OUTPUT_PATH.empty()) {
            result.save_image(OUTPUT_PATH.c_str());
        }
    } else {
        // Run the parallel convolution.
        if (MEMORY_TYPE == "global") {
            // Run the global memory convolution.
            Image result = Parallel::Convolution::convolve_global(image, kernel, PADDING_TYPE, RESULTS_PATH);

            // Save the convolved image.
            if (!OUTPUT_PATH.empty()) {
                result.save_image(OUTPUT_PATH.c_str());
            }
        } else if (MEMORY_TYPE == "constant") {
            // Run the constant memory convolution.
            Image result = Parallel::Convolution::convolve_constant(image, kernel, PADDING_TYPE, RESULTS_PATH);

            // Save the convolved image.
            if (!OUTPUT_PATH.empty()) {
                result.save_image(OUTPUT_PATH.c_str());
            }
        } else if (MEMORY_TYPE == "shared") {
            // Run the shared memory convolution.
            Image result = Parallel::Convolution::convolve_shared(image, kernel, PADDING_TYPE, RESULTS_PATH);

            // Save the convolved image.
            if (!OUTPUT_PATH.empty()) {
                result.save_image(OUTPUT_PATH.c_str());
            }
        } else {
            // Run the pinned memory convolution.
            Image result = Parallel::Convolution::convolve_pinned(image, kernel, PADDING_TYPE, RESULTS_PATH, 3);

            // Save the convolved image.
            if (!OUTPUT_PATH.empty()) {
                result.save_image(OUTPUT_PATH.c_str());
            }
        }
    }    
}


int main(int argc, char* argv[]) {
    // Process the input.
    if (processInput(argc, argv) != 0) {
        return 1;
    }

    // Load the image.
    Image image(IMAGE_PATH.c_str(), 0, SOA);

    // Load the kernel and run the convolution.
    if (KERNEL == "box_blur") {
        Kernel kernel = Kernel::box_blur_kernel();
        runConvolution(image, kernel);
    } else if (KERNEL == "gaussian_blur") {
        Kernel kernel = Kernel::gaussian_blur_kernel();
        runConvolution(image, kernel);
    } else if (KERNEL == "sharpen") {
        Kernel kernel = Kernel::sharpen_kernel();
        runConvolution(image, kernel);
    } else if (KERNEL == "edge_detection") {
        Kernel kernel = Kernel::edge_detection_kernel();
        runConvolution(image, kernel);
    } else if (KERNEL == "unsharpen_mask") {
        Kernel kernel = Kernel::unsharpen_mask_kernel();
        runConvolution(image, kernel);
    } else if (KERNEL == "emboss") {
        Kernel kernel = Kernel::emboss_kernel();
        runConvolution(image, kernel);
    } else {
        Kernel kernel = Kernel::custom_kernel(KERNEL_SIZE, KERNEL_DATA, KERNEL_NORMALIZATION);
        runConvolution(image, kernel);
    }

    return 0;
}
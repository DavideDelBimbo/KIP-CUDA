#include <sstream>
#include <iomanip>
#include <iostream>

#include "kernel.h"


// Constructor and destructor.

Kernel::Kernel(const int width, const int height) : width(width), height(height)
{
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Kernel dimensions must be greater than 0." << std::endl;
        throw std::invalid_argument("Kernel dimensions must be greater than 0.");
    }

    if (width % 2 == 0 || height % 2 == 0) {
        std::cerr << "Error: Kernel dimensions must be odd." << std::endl;
        throw std::invalid_argument("Kernel dimensions must be odd.");
    }

    if (width != height) {
        std::cerr << "Error: Kernel dimensions must be equal." << std::endl;
        throw std::invalid_argument("Kernel dimensions must be equal.");
    }

    // Calculate the size of the kernel.
    size_t size = get_size();

    // Allocate memory for the kernel.
    data = new float[size]{0};
}

Kernel::Kernel(const int width, const int height, float *data) : Kernel(width, height) {
    // Get the size of the kernel.
    size_t size = get_size();

    // Copy the kernel data.
    memcpy(this->data, data, size * sizeof(float));
}

Kernel::~Kernel() {
    // Free the kernel data.
    delete[] data;
}


// Getters.

int Kernel::get_width() const {
    return width;
}

int Kernel::get_height() const {
    return height;
}

size_t Kernel::get_size() const {
    return (size_t)(width * height);
}

float* Kernel::get_data() const {
    return data;
}


// Predefined kernels.

Kernel Kernel::gaussian_blur_kernel() {
    float data[9] = {
        1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16
    };

    return Kernel(3, 3, data);
}

Kernel Kernel::box_blur_kernel() {
    float data[9] = {
        1.0/9, 1.0/9, 1.0/9,
        1.0/9, 1.0/9, 1.0/9,
        1.0/9, 1.0/9, 1.0/9
    };

    return Kernel(3, 3, data);
}

Kernel Kernel::edge_detection_kernel() {
    float data[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    return Kernel(3, 3, data);
}

Kernel Kernel::sharpen_kernel() {
    float data[9] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };

    return Kernel(3, 3, data);
}

Kernel Kernel::unsharpen_mask_kernel() {
    float data[25] = {
        -1.0/256, -4.0/256, -6.0/256, -4.0/256, -1.0/256,
        -4.0/256, -16.0/256, -24.0/256, -16.0/256, -4.0/256,
        -6.0/256, -24.0/256, 476.0/256, -24.0/256, -6.0/256,
        -4.0/256, -16.0/256, -24.0/256, -16.0/256, -4.0/256,
        -1.0/256, -4.0/256, -6.0/256, -4.0/256, -1.0/256
    };

    return Kernel(5, 5, data);
}

Kernel Kernel::emboss_kernel() {
    float data[9] = {
        -2, -1, 0,
        -1,  1, 1,
        0,  1, 2
    };

    return Kernel(3, 3, data);
}


// Custom kernel.

Kernel Kernel::custom_kernel(const int size, float* data, const bool normalize) {
    // Normalize the kernel.
    if (normalize) {
        // Normalize the data array.
        float sum = 0.0;
        for (int i = 0; i < size * size; i++) {
            sum += data[i];
        }

        for (int i = 0; i < size * size; i++) {
            data[i] /= sum;
        }
    }

    return Kernel(size, size, data);
}


// Operators.

float &Kernel::operator()(const int col, const int row) const {
    // Check if the coordinates are valid.
    if ((col < 0 || col >= width) || (row < 0 || row >= height)) {
        std::cerr << "Error: Invalid kernel coordinates (" << col << ", " << row << ")." << std::endl;
        throw std::invalid_argument("Invalid kernel coordinates.");
    }

    // Get the 1D kernel index.
    const int kernel_index = (row * width) + col;

    return data[kernel_index];
}

std::ostream &operator<<(std::ostream &os, const Kernel &kernel) {
    // Calculate the maximum element width.
    std::string str;
    int maxElementWidth = 1;
    for (int i = 0; i < kernel.get_size(); i++) {
        // Get the string representation of the element.
        str = std::to_string(kernel.get_data()[i]);

        // Remove trailing zeros.
        str.erase(str.find_last_not_of('0') + 1, std::string::npos);

        // Remove trailing decimal point.
        str.erase (str.find_last_not_of('.') + 1, std::string::npos);

        // Update the maximum element width.
        maxElementWidth = std::max(maxElementWidth, (int)str.size());
    }

    // Print the kernel dimensions.
    os << "Kernel dimensions: " << kernel.width << "x" << kernel.height << std::endl;

    // Print the kernel data.
    os << "Kernel data: " << std::endl;
    for (int row = 0; row < kernel.get_height(); row++) {
        for (int col = 0; col < kernel.get_width(); col++) {
            os << std::setw(maxElementWidth) << (float)kernel(col, row) << " ";
        }
        os << std::endl;
    }

    // Restore the output format
    os << std::defaultfloat;

    return os;
}
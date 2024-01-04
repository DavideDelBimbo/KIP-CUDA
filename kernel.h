#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>
#include <cstdio>


class Kernel {
    public:
        // Constructor and destructor.

        /*
            * Create a kernel with the given dimensions.
            *
            * @param width The width of the kernel.
            * @param height The height of the kernel.
        */
        Kernel(const int width, const int height);

        /*
            * Create a kernel with the given dimensions and data.
            *
            * @param width The width of the kernel.
            * @param height The height of the kernel.
            * @param data The data of the kernel.
        */
        Kernel(const int width, const int height, float *data);

        /*
            * Destructor.
        */
        ~Kernel();


        // Getters.

        /*
            * Get the kernel width.
            *
            * @return The kernel width.
        */
        int get_width() const;

        /*
            * Get the kernel height.
            *
            * @return The kernel height.
        */
        int get_height() const;

        /*
            * Get the kernel size.
            *
            * @return The kernel size.
        */
        size_t get_size() const;

        /*
            * Get the kernel data.
            *
            * @return The kernel data.
        */
        float* get_data() const;


        // Predefined kernels.

        /*
            * Create a gaussian blur kernel.
            *
            * @return The gaussian blur kernel.
        */
        static Kernel gaussian_blur_kernel();

        /*
            * Create a box blur kernel.
            * 
            * @return The box blur kernel.
        */
        static Kernel box_blur_kernel();

        /*
            * Create a edge detection kernel.
            *
            * @return The edge detection kernel.
        */
        static Kernel edge_detection_kernel();

        /*
            * Create a sharpen kernel.
            *
            * @return The sharpen kernel.
        */
        static Kernel sharpen_kernel();

        /*
            * Create a unsharpen mask kernel.
            *
            * @return The unsharpen mask kernel.
        */
        static Kernel unsharpen_mask_kernel();

        /*
            * Create a emboss kernel.
            *
            * @return The emboss kernel.
        */
        static Kernel emboss_kernel();


        // Custom kernel.

        /*
            * Create a custom kernel.
            *
            * @param size The size of the kernel.
            * @param data The data of the kernel.
            * @param normalize Normalize the kernel.
            * 
            * @return The custom kernel.
        */
        static Kernel custom_kernel(const int size, float *data, const bool normalize = true);


        // Operators.

        /*
            * Get the kernel value at the given position.
            *
            * @param col The column of the kernel value.
            * @param row The row of the kernel value.
            *
            * @return The kernel value at the given position.
        */
        float& operator()(const int col, const int row) const;
        
        /*
            * Print the kernel.
        */
        friend std::ostream& operator<<(std::ostream& os, const Kernel& kernel);


    private:
        // Attributes.

        // Kernel dimensions.
        int width = 0, height = 0;
        // Kernel data.
        float *data = NULL;
};

#endif // KERNEL_H
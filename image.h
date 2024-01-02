#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>
#include <cstdio>
#include <stdexcept>


// Image types.
enum ImageType {
    PNG,
    JPG,
    JPEG,
    BMP,
    TGA,
    UNKNOWN
};


// Padding types.
enum PaddingType {
    ZERO,
    REPLICATE,
    MIRROR
};


class Image {
    public:
        // Constructors and destructor.

        /*
            * Load an image from a file.
            *
            * @param filename The name of the file.
            * @param channel_force The number of channels to force the image to have (default: 0).
            * @param is_SoA Whether the image is in SoA architecture (default: false).
        */
        Image(const char* filename, const int channel_force = 0, const bool is_SoA = false);

        /*
            * Create an empty image with the given dimensions.
            *
            * @param width The width of the image.
            * @param height The height of the image.
            * @param channels The number of channels of the image.
            * @param is_SoA Whether the image is in SoA architecture (default: false).
        */
        Image(const int width, const int height, const int channels, const bool is_SoA = false);

        /*
            * Create an image with the given dimensions and data.
            *
            * @param width The width of the image.
            * @param height The height of the image.
            * @param channels The number of channels of the image.
            * @param data The data to fill the image with.
            * @param is_SoA Whether the image is in SoA architecture (default: false).
        */
        Image(const int width, const int height, const int channels, uint8_t* data, const bool is_SoA = false);
        
        /*
            * Copy constructor for an image.
            *
            * @param image The image to be copied.
        */
        Image(const Image& image);

        /*
            * Destructor.
        */
        ~Image();


        // Getters.

        /*
            * Get the width of the image.
            *
            * @return The width of the image.  
        */
        int get_width() const;
        
        /*
            * Get the height of the image.
            *
            * @return The height of the image.
        */
        int get_height() const;
        
        /*
            * Get the number of channels of the image.
            *
            * @return The number of channels of the image.
        */
        int get_channels() const;
        
        /*
            * Get the size of the image.
            *
            * @return The size of the image.
        */
        size_t get_size() const;
        
        /*
            * Get the linearized data of the image.
            *
            * @return The linearized data of the image.
        */
        uint8_t* get_data() const;

        /*
            * Get architecture of the image.
            *
            * @return True if the image is in SoA architecture, false otherwise.
        */
        bool get_is_SoA() const;

        /*
            * Get the type of an image from its filename.
            *
            * @param filename The name of the file.
            * 
            * @return The type of the image.
        */
        ImageType get_image_type(const char* filename) const;


        // Methods.

        /*
            * Load an image from a filename path.
            *
            * @param filename The path of the image to be loaded.
            * @param channel_force The number of channels to force the image to have.
            * 
            * @return True if the image was loaded successfully, false otherwise.
        */
        bool load_image(const char* filename, const int channel_force = 0);

        /*
            * Save an image to a filename path.
            *
            * @param filename The path of the image to be saved.
            * @param AoS Whether to save the image in AoS or SoA architecture.
        */
        void save_image(const char* filename);

        /*
            * Applies padding to the image.
            *
            * @param image The image to be padded.
            * @param padding_width The padding width.
            * @param padding_height The padding height.
            * @param padding_type The padding type.
            * 
            * @return The padded image.
        */
        Image padding(const int padding_width, const int padding_height, const PaddingType padding_type) const;


        // Operators.

        /*
            * Assignment operator for an image.
            *
            * @param other The image to be assigned.
        */
        Image& operator=(const Image& other);

        /*
            * Get the image pixel value at the given position.
            *
            * @param col The column of the image pixel.
            * @param row The row of the image pixel.
            * @param channel The channel of the image pixel.
            * 
            * @return The image pixel value at the given position.
        */
        uint8_t& operator()(const int col, const int row, const int channel) const;

        /*
            * Compare two images.
            *
            * @param other The image to be compared.
            * 
            * @return True if the images are equal, false otherwise.
        */
        bool operator==(const Image& other) const;

        /*
            * Print the image.
        */
        friend std::ostream& operator<<(std::ostream& os, const Image& image);

        
    private:
        // Attributes.

        // Image dimensions.
        int width = 0, height = 0, channels = 0;

        // Image data.
        uint8_t* data = NULL;
        
        // Flag to specify whether the image is in SoA architecture.
        bool is_SoA = false;


        // Methods.

        /*
            * Convert the image data from AoS to SoA architecture.
        */
        void AoS_to_SoA();

        /*
            * Convert the image data from SoA to AoS architecture.
        */
        void SoA_to_AoS();
};

#endif // IMAGE_H
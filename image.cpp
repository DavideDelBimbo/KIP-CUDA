#include <sstream>
#include <iomanip>

#include "image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define clamp(start, x, end) std::min(std::max(start, x), end)


// Constructors and destructor.

Image::Image(const char *filename, const int channel_force, const bool is_SoA) : is_SoA(is_SoA) {
    // Load the image.
    if (!load_image(filename, channel_force)) {
        throw std::runtime_error("Failed to read " + std::string(filename) + ".");
    } else {
        printf("Read %s:\n\tWidth: %dpx\n\tHeight: %dpx\n\tChannels: %d\n\n", filename, width, height, channels);
    }
}

Image::Image(const int width, const int height, const int channels, const bool is_SoA) : width(width), height(height), channels(channels), is_SoA(is_SoA) {
    // Get the size of the image.
    size_t size = get_size();
    
    // Allocate memory for the image.
    data = new uint8_t[size]{0};
}

Image::Image(const int width, const int height, const int channels, uint8_t* data, const bool is_SoA) : Image(width, height, channels, is_SoA) {
    // Get the size of the image.
    size_t size = get_size();
    
    // Copy the image data.
    memcpy(this->data, data, size * sizeof(uint8_t));
}

Image::Image(const Image &image) : Image(image.width, image.height, image.channels, image.is_SoA) {
    // Get the size of the image.
    size_t size = get_size();

    // Copy the image data.
    memcpy(data, image.data, size * sizeof(uint8_t));
}

Image::~Image() {
    // Free the image data.
    delete[] data;
}


// Getters.

int Image::get_width() const {
    return width;
}

int Image::get_height() const {
    return height;
}

int Image::get_channels() const {
    return channels;
}

size_t Image::get_size() const {
    return (size_t)(width * height * channels);
}

uint8_t* Image::get_data() const {
    return data;
}

bool Image::get_is_SoA() const {
    return is_SoA;
}

ImageType Image::get_image_type(const char *filename) const {
    // Get the file extension.
    const char* extension = strrchr(filename, '.');

    // Check if the extension is valid.
    if (extension != NULL) {
        if (strcmp(extension, ".png") == 0)
            return ImageType::PNG;
        else if (strcmp(extension, ".jpg") == 0)
            return ImageType::JPG;
        else if (strcmp(extension, ".bmp") == 0)
            return ImageType::BMP;
        else if (strcmp(extension, ".tga") == 0)
            return ImageType::TGA;
    }

    return ImageType::UNKNOWN;
}


// Methods.

bool Image::load_image(const char *filename, const int channel_force) {
    // Load the image.
    uint8_t* loaded_data = stbi_load(filename, &width, &height, &channels, channel_force);

    // Check if the image was loaded successfully.
    if (loaded_data != NULL) {
        // Force the number of channels.
        channels = (channel_force == 0) ? channels : channel_force;

        // Get the size of the image.
        size_t size = get_size();

        // Allocate memory for the image.
        data = new uint8_t[size]{0};

        // Copy the image data.
        memcpy(data, loaded_data, get_size() * sizeof(uint8_t));

        // Convert to SoA if required.
        if (is_SoA) { AoS_to_SoA(); }
    }

    // Free the loaded image.
    stbi_image_free(loaded_data);

    return data != NULL;
}

void Image::save_image(const char* filename) {
    // Convert to AoS if required.
    if (is_SoA) { SoA_to_AoS(); }

    // Get the image type.
    ImageType type = get_image_type(filename);

    // Save the image.
    if (type == ImageType::PNG)
        stbi_write_png(filename, width, height, channels, data, width * channels);
    else if (type == ImageType::JPG || type == ImageType::JPEG)
        stbi_write_jpg(filename, width, height, channels, data, 100);
    else if (type == ImageType::BMP)
        stbi_write_bmp(filename, width, height, channels, data);
    else if (type == ImageType::TGA)
        stbi_write_tga(filename, width, height, channels, data);
    else
        printf("Failed to save %s\n", filename);
}

Image Image::padding(const int padding_width, const int padding_height, const PaddingType padding_type) const {
    // Check if the padding dimensions are valid.
    if (padding_width < 0 || padding_height < 0) {
        throw std::invalid_argument("Invalid padding dimensions.");
    }
    
    // Calculate the padded dimensions.
    const int padded_width = width + (padding_width * 2); // Padded width.
    const int padded_height = height + (padding_height * 2); // Padded height.

    // Create the padded image.
    Image padded_image(padded_width, padded_height, channels, is_SoA);

    // Iterate over the input image.
    for (int channel = 0; channel < channels; channel++) {
        for (int y = 0; y < padded_height; y++) {
            for (int x = 0; x < padded_width; x++) {
                // Get the pixel index to be padded.
                const int col = x - padding_width; // Column index.
                const int row = y - padding_height; // Row index.
                
                // Verify if the pixel index is valid.
                if (col >= 0 && col < width && row >= 0 && row < height) {
                    // Set the padded pixel to the input pixel.
                    padded_image(x, y, channel) = (*this)(col, row, channel);
                } else {
                    if (padding_type == PaddingType::ZERO) {
                        // Set the padded pixel to zero.
                        padded_image(x, y, channel) = 0;
                    } else if (padding_type == PaddingType::REPLICATE) {
                        // Get the nearest valid pixel.
                        const int valid_col = clamp(0, col, width - 1); // Column index of the nearest valid pixel.
                        const int valid_row = clamp(0, row, height - 1); // Row index of the nearest valid pixel.

                        // Set the padded pixel to the nearest valid pixel.
                        padded_image(x, y, channel) = (*this)(valid_col, valid_row, channel);
                    } else if (padding_type == PaddingType::MIRROR) {
                        // Get the mirrored pixel.
                        int mirror_col = std::abs(col % (2 * width)); // Column index of the mirrored pixel.
                        int mirror_row = std::abs(row % (2 * height)); // Row index of the mirrored pixel.

                        // Ensure that the mirrored pixel is in the valid range.
                        mirror_col = std::min(mirror_col, ((2 * width) - 1) - (mirror_col + 1));
                        mirror_row = std::min(mirror_row, ((2 * height) - 1) - (mirror_row + 1));

                        // Set the padded pixel to the mirrored pixel.
                        padded_image(x, y, channel) = (*this)(mirror_col, mirror_row, channel);
                    }
                }
            }
        }
    }

    return padded_image;
}


// Operators.

Image &Image::operator=(const Image &other) {
    // Check if the images are different.
    if (this != &other) {
        // Get the size of the image.
        size_t size = get_size();

        // Copy the image data.
        memcpy(data, other.data, size * sizeof(uint8_t));
    }

    return *this;
}

uint8_t &Image::operator()(const int col, const int row, const int channel) const {
    // Check if the coordinates are valid.
    if ((col < 0 || col >= width) || (row < 0 || row >= height) || (channel < 0 || channel >= channels)) {
        throw std::invalid_argument("Invalid coordinates.");
    }

    // Get the 1D pixel index.
    const int pixel_index = is_SoA ? ((channel * width * height) + (row * width) + col) : ((row * width + col) * channels + channel);

    return data[pixel_index];
}

bool Image::operator==(const Image &other) const {
    // Check if the images have the same dimensions.
    if (width != other.width || height != other.height || channels != other.channels) {
        return false;
    }

    // Get the size of the image.
    size_t size = get_size();

    // Compare the image data.
    return memcmp(data, other.data, size * sizeof(uint8_t)) == 0;
}

std::ostream &operator<<(std::ostream &os, const Image &image) {
    // Calculate the maximum element width.
    std::string str;
    int maxElementWidth = 1;
    for (int i = 0; i < image.get_size(); i++) {
        // Get the string representation of the element.
        str = std::to_string(image.get_data()[i]);

        // Update the maximum element width.
        maxElementWidth = std::max(maxElementWidth, (int)str.size());
    }

    // Print the image data.
    os << "Image data: " << std::endl;
    for (int row = 0; row < image.get_height(); row++) {
        for (int col = 0; col < image.get_width(); col++) {
            os << "(";
            for (int channel = 0; channel < image.get_channels(); channel++) {
                os << std::setw(maxElementWidth) << (int)image(col, row, channel);
                if (channel < image.get_channels() - 1) {
                    os << ", ";
                }
            }
            os << ") ";
        }
        os << std::endl;
    }

    // Restore the output format
    os << std::defaultfloat;

    return os;
}


// Private methods.

void Image::AoS_to_SoA() {
    // Get the size of the image.
    size_t size = get_size();

    // Allocate memory for the image in SoA architecture.
    uint8_t* data_SoA = new uint8_t[size]{0};

    // Copy the image data in SoA architecture.
    for (int channel = 0; channel < channels; channel++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {                
                data_SoA[channel * width * height + row * width + col] = data[(row * width + col) * channels + channel];
            }
        }
    }

    // Update the existing data array.
    delete[] data;
    data = data_SoA; 
}

void Image::SoA_to_AoS() {
    // Get the size of the image.
    size_t size = get_size();

    // Allocate memory for the image in AoS architecture.
    uint8_t* data_AoS = new uint8_t[size]{0};

    // Copy the image data in AoS architecture.
    for (int channel = 0; channel < channels; channel++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {                
                data_AoS[(row * width + col) * channels + channel] = data[channel * width * height + row * width + col];
            }
        }
    }

    // Update the existing data array.
    delete[] data;
    data = data_AoS;
}
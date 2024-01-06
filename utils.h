#ifndef K_UTILS_H
#define K_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>


#include "params.h"

/*
    * Function to save results to a file.
    *
    * @param basePath The base path to save the results.
    * @param executionType The execution type.
    * @param image_width The image width.
    * @param image_height The image height.
    * @param image_channels The image channels.
    * @param image_is_SoA The image architecture.
    * @param kernel_width The kernel width.
    * @param kernel_height The kernel height.
    * @param copyTime The copy time.
    * @param executionTime The execution time.
    * @param iterations The number of iterations.
*/ 
inline void save_results(const std::string& base_path, std::string& execution_type, int image_width, int image_height, int image_channels, int image_is_SoA, int kernel_width, int kernel_height, float execution_time, int iterations) {
    struct stat buffer;
    std::ofstream outfile;


    // Convert to lowercase the execution type string.
    std::transform(execution_type.begin(), execution_type.end(), execution_type.begin(), ::tolower);


    // Check if the file exists
    if (stat((base_path + "results.txt").c_str(), &buffer) == 0) {
        // File exists, append to existing one
        outfile.open(base_path + "results.txt", std::ios_base::app);
    } else {
        // File doesn't exist, create new one with header
        outfile.open(base_path + "results.txt");
        outfile << "execution_type,image_width,image_height,image_channels,image_architecture,kernel_width,kernel_height,execution_time,iterations" << std::endl;
        
    }

    // Save the results.
    outfile << execution_type << "," << image_width << "," << image_height << "," << image_channels << "," << (image_is_SoA ? "SoA" : "AoS") << "," << kernel_width << "," << kernel_height << "," << execution_time << "," << iterations << std::endl;
    outfile.close();
}

#endif // K_UTILS_H
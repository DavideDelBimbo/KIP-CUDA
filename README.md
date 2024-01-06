# KIP-CUDA
This repository contains a C++ implementation of the convolution operation for Kernel Image Processing, optimized for parallel execution through CUDA. The integration of CUDA to enable parallelization on GPUs allows for significant performance improvements over sequential implementations.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Prerequisites
Before you get started, make sure you have the following dependencies installed:

- C++ compiler with OpenMP support (e.g. g++).
- CUDA compiler (nvcc).

## Installation
Follow these steps to set up and compile the code:
1. Clone this repository to your local machine using the following command:
<p align="center"><code>git clone DavideDelBimbo/KIP-CUDA</code></p>

2. Navigate to the project directory:
<p align="center"><code>cd KIP-CUDA</code></p>

3. Modify the parameters in `params.h` as needed to customize the behavior of the application.

4. Compile the code using nvcc:
<p align="center"><code>nvcc main.cu image.cpp kernel.cpp parallel/convolution.cu sequential/convolution.cpp -o kip</code></p>

## Usage
To execute the code, use the following command:
<p align="center"><code>./kip --image_path --SoA --padding_type --kernel [--kernel_size --kernel_data --kernel_normalization] --execution_type [--memory_type] --output_path --results_path</code></p>

Where:
- `--image_path`: Path to the original input image file.
- `--SoA` (optional): Convert image to SoA (Structure of Arrays) architecture.
- `--padding_type` (optional): Type of padding to be applied to the input image (`zero`, `replicate` or `mirror`). Default is `mirror`.
- `--kernel`: Type of kernel to be convolved with the input image (`box_blur`, `gaussian_blur`, `sharpen`, `edge_detection`, `unsharpen_mask`, `emboss` or `custom`).
- `--kernel-size` (required only with `<kernel> = 'custom'`): Size of custom kernel.
- `--kernel-data` (required only with `<kernel> = 'custom'`): Data of custom kernel.
- `--kernel-normalization` (optional with `<kernel> = 'custom'`): Normalize kernel data.
- `--execution_type`: The execution type (use either `parallel` or `sequential`).
- `--memory_type` (required only with `<execution_type> = 'parallel'`): Level of memory to use for convolution (`global`, `constant`, `shared` or `pinned`).
- `--output_path` (optional): Path to the output image file.
- `--results_path` (optional): Base path for the results (default: `./results/`).

For example:
main.exe -I="images/480.jpg" -P="mirror" -K="gaussian_blur" -E="sequential" -O="results/images/resolutions/480/480_sequential_gaussianBlur.jpg" -R=".\results\"

<p align="center"><code>./kip --image_path='images/480.jpg' --padding_type='mirror' --kernel='gaussian_blur' --execution_type='sequential' --output_path='./results/images/480_gaussianBlur.jpg' --base_path='./results/'</code></p>
<p align="center"><code>./kip --image_path='images/480.jpg' --SoA --padding_type='mirror' --kernel='gaussian_blur' --execution_type='parallel' --memory_type='global' --output_path='./results/images/480_gaussianBlur.jpg' --base_path='./results/'</code></p>

## Results
The results obtained from running the convolution operation using CUDA can be found in <a href="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/report.pdf" target="_blank">report</a> file. The results may include information such as the output convolved images, execution times and any relevant statistics.

Before diving into the results, you may want to take a moment to configure the parameters in `params.h` to customize the behavior of the convolution opeartion according to your specific needs.

<p float="left" align="center">
    <p float="left" align="center">
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/original.png" alt="Original image" width="200" />
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/box_blur.png" alt="Box blur image" width="200" />
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/gaussian_blur.png" alt="Gaussian blur image" width="200" />
    </p>
    <p float="left" align="center">
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/edge_detection.png" alt="Edge detection image" width="200" />
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/sharpen.png" alt="Sharpen image" width="200" />
        <img src="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/report/unsharp_mask.png" alt="Unsharp mask image" width="200" />
    </p>
</p>

## License
This project is licensed under the <a href="https://github.com/DavideDelBimbo/KIP-CUDA/blob/main/LICENSE" target="_blank">MIT</a> License.
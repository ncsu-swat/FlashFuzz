#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract pooling parameters first
        uint8_t kernel_size = Data[offset++] % 5 + 1;
        uint8_t stride = Data[offset++] % 5 + 1;
        uint8_t padding = Data[offset++] % 3;
        uint8_t dilation = Data[offset++] % 3 + 1;
        bool ceil_mode = Data[offset++] % 2 == 0;
        bool count_include_pad = Data[offset++] % 2 == 0;
        bool return_indices = Data[offset++] % 2 == 0;
        uint8_t pool_type_selector = Data[offset++];
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        int64_t dim = input.dim();
        
        // Adaptive output size
        int64_t output_size = 1 + (pool_type_selector % 5);
        
        // MaxPool1d - requires 2D (unbatched) or 3D (batched) input
        if (dim == 2 || dim == 3) {
            try {
                auto options = torch::nn::MaxPool1dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode);
                
                auto max_pool1d = torch::nn::MaxPool1d(options);
                
                if (return_indices) {
                    auto result = max_pool1d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = max_pool1d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // MaxPool2d - requires 3D (unbatched) or 4D (batched) input
        if (dim == 3 || dim == 4) {
            try {
                auto options = torch::nn::MaxPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode);
                
                auto max_pool2d = torch::nn::MaxPool2d(options);
                
                if (return_indices) {
                    auto result = max_pool2d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = max_pool2d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // MaxPool3d - requires 4D (unbatched) or 5D (batched) input
        if (dim == 4 || dim == 5) {
            try {
                auto options = torch::nn::MaxPool3dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .dilation(dilation)
                    .ceil_mode(ceil_mode);
                
                auto max_pool3d = torch::nn::MaxPool3d(options);
                
                if (return_indices) {
                    auto result = max_pool3d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = max_pool3d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // AvgPool1d - requires 2D or 3D input
        if (dim == 2 || dim == 3) {
            try {
                auto options = torch::nn::AvgPool1dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad);
                
                auto avg_pool1d = torch::nn::AvgPool1d(options);
                auto output = avg_pool1d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // AvgPool2d - requires 3D or 4D input
        if (dim == 3 || dim == 4) {
            try {
                auto options = torch::nn::AvgPool2dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad);
                
                auto avg_pool2d = torch::nn::AvgPool2d(options);
                auto output = avg_pool2d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // AvgPool3d - requires 4D or 5D input
        if (dim == 4 || dim == 5) {
            try {
                auto options = torch::nn::AvgPool3dOptions(kernel_size)
                    .stride(stride)
                    .padding(padding)
                    .ceil_mode(ceil_mode)
                    .count_include_pad(count_include_pad);
                
                auto avg_pool3d = torch::nn::AvgPool3d(options);
                auto output = avg_pool3d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // AdaptiveMaxPool1d - requires 2D or 3D input
        if (dim == 2 || dim == 3) {
            try {
                auto options = torch::nn::AdaptiveMaxPool1dOptions(output_size);
                auto adaptive_max_pool1d = torch::nn::AdaptiveMaxPool1d(options);
                
                if (return_indices) {
                    auto result = adaptive_max_pool1d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = adaptive_max_pool1d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // AdaptiveMaxPool2d - requires 3D or 4D input
        if (dim == 3 || dim == 4) {
            try {
                auto options = torch::nn::AdaptiveMaxPool2dOptions({output_size, output_size});
                auto adaptive_max_pool2d = torch::nn::AdaptiveMaxPool2d(options);
                
                if (return_indices) {
                    auto result = adaptive_max_pool2d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = adaptive_max_pool2d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // AdaptiveMaxPool3d - requires 4D or 5D input
        if (dim == 4 || dim == 5) {
            try {
                auto options = torch::nn::AdaptiveMaxPool3dOptions({output_size, output_size, output_size});
                auto adaptive_max_pool3d = torch::nn::AdaptiveMaxPool3d(options);
                
                if (return_indices) {
                    auto result = adaptive_max_pool3d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = adaptive_max_pool3d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // AdaptiveAvgPool1d - requires 2D or 3D input
        if (dim == 2 || dim == 3) {
            try {
                auto options = torch::nn::AdaptiveAvgPool1dOptions(output_size);
                auto adaptive_avg_pool1d = torch::nn::AdaptiveAvgPool1d(options);
                auto output = adaptive_avg_pool1d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // AdaptiveAvgPool2d - requires 3D or 4D input
        if (dim == 3 || dim == 4) {
            try {
                auto options = torch::nn::AdaptiveAvgPool2dOptions({output_size, output_size});
                auto adaptive_avg_pool2d = torch::nn::AdaptiveAvgPool2d(options);
                auto output = adaptive_avg_pool2d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // AdaptiveAvgPool3d - requires 4D or 5D input
        if (dim == 4 || dim == 5) {
            try {
                auto options = torch::nn::AdaptiveAvgPool3dOptions({output_size, output_size, output_size});
                auto adaptive_avg_pool3d = torch::nn::AdaptiveAvgPool3d(options);
                auto output = adaptive_avg_pool3d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // FractionalMaxPool2d - requires 3D or 4D input
        if (dim == 3 || dim == 4) {
            try {
                int64_t frac_output_size = output_size + 1;
                auto options = torch::nn::FractionalMaxPool2dOptions({frac_output_size, frac_output_size});
                auto fractional_max_pool2d = torch::nn::FractionalMaxPool2d(options);
                
                if (return_indices) {
                    auto result = fractional_max_pool2d->forward_with_indices(input);
                    (void)std::get<0>(result);
                    (void)std::get<1>(result);
                } else {
                    auto output = fractional_max_pool2d->forward(input);
                    (void)output;
                }
            } catch (...) {}
        }
        
        // LPPool1d - requires 2D or 3D input
        if (dim == 2 || dim == 3) {
            try {
                double norm_type = 1.0 + (pool_type_selector % 3);
                auto options = torch::nn::LPPool1dOptions(norm_type, kernel_size)
                    .stride(stride)
                    .ceil_mode(ceil_mode);
                
                auto lp_pool1d = torch::nn::LPPool1d(options);
                auto output = lp_pool1d->forward(input);
                (void)output;
            } catch (...) {}
        }
        
        // LPPool2d - requires 3D or 4D input
        if (dim == 3 || dim == 4) {
            try {
                double norm_type = 1.0 + (pool_type_selector % 3);
                auto options = torch::nn::LPPool2dOptions(norm_type, kernel_size)
                    .stride(stride)
                    .ceil_mode(ceil_mode);
                
                auto lp_pool2d = torch::nn::LPPool2d(options);
                auto output = lp_pool2d->forward(input);
                (void)output;
            } catch (...) {}
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
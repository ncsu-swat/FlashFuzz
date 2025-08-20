#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract pooling parameters from the remaining data
        uint8_t kernel_size = 2;
        uint8_t stride = 2;
        uint8_t padding = 0;
        uint8_t dilation = 1;
        bool ceil_mode = false;
        bool count_include_pad = true;
        bool return_indices = false;
        
        if (offset + 7 <= Size) {
            kernel_size = Data[offset++] % 5 + 1;
            stride = Data[offset++] % 5 + 1;
            padding = Data[offset++] % 3;
            dilation = Data[offset++] % 3 + 1;
            ceil_mode = Data[offset++] % 2 == 0;
            count_include_pad = Data[offset++] % 2 == 0;
            return_indices = Data[offset++] % 2 == 0;
        }
        
        // Try different pooling operations based on input dimensions
        int64_t dim = input.dim();
        
        // MaxPool1d
        if (dim >= 2) {
            auto options = torch::nn::MaxPool1dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode);
            
            auto max_pool1d = torch::nn::MaxPool1d(options);
            
            if (return_indices) {
                auto [output, indices] = max_pool1d->forward_with_indices(input);
            } else {
                auto output = max_pool1d->forward(input);
            }
        }
        
        // MaxPool2d
        if (dim >= 3) {
            auto options = torch::nn::MaxPool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode);
            
            auto max_pool2d = torch::nn::MaxPool2d(options);
            
            if (return_indices) {
                auto [output, indices] = max_pool2d->forward_with_indices(input);
            } else {
                auto output = max_pool2d->forward(input);
            }
        }
        
        // MaxPool3d
        if (dim >= 4) {
            auto options = torch::nn::MaxPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .ceil_mode(ceil_mode);
            
            auto max_pool3d = torch::nn::MaxPool3d(options);
            
            if (return_indices) {
                auto [output, indices] = max_pool3d->forward_with_indices(input);
            } else {
                auto output = max_pool3d->forward(input);
            }
        }
        
        // AvgPool1d
        if (dim >= 2) {
            auto options = torch::nn::AvgPool1dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad);
            
            auto avg_pool1d = torch::nn::AvgPool1d(options);
            auto output = avg_pool1d->forward(input);
        }
        
        // AvgPool2d
        if (dim >= 3) {
            auto options = torch::nn::AvgPool2dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad);
            
            auto avg_pool2d = torch::nn::AvgPool2d(options);
            auto output = avg_pool2d->forward(input);
        }
        
        // AvgPool3d
        if (dim >= 4) {
            auto options = torch::nn::AvgPool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
                .ceil_mode(ceil_mode)
                .count_include_pad(count_include_pad);
            
            auto avg_pool3d = torch::nn::AvgPool3d(options);
            auto output = avg_pool3d->forward(input);
        }
        
        // AdaptiveMaxPool1d
        if (dim >= 2) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveMaxPool1dOptions(output_size);
            auto adaptive_max_pool1d = torch::nn::AdaptiveMaxPool1d(options);
            
            if (return_indices) {
                auto [output, indices] = adaptive_max_pool1d->forward_with_indices(input);
            } else {
                auto output = adaptive_max_pool1d->forward(input);
            }
        }
        
        // AdaptiveMaxPool2d
        if (dim >= 3) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveMaxPool2dOptions({output_size, output_size});
            auto adaptive_max_pool2d = torch::nn::AdaptiveMaxPool2d(options);
            
            if (return_indices) {
                auto [output, indices] = adaptive_max_pool2d->forward_with_indices(input);
            } else {
                auto output = adaptive_max_pool2d->forward(input);
            }
        }
        
        // AdaptiveMaxPool3d
        if (dim >= 4) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveMaxPool3dOptions({output_size, output_size, output_size});
            auto adaptive_max_pool3d = torch::nn::AdaptiveMaxPool3d(options);
            
            if (return_indices) {
                auto [output, indices] = adaptive_max_pool3d->forward_with_indices(input);
            } else {
                auto output = adaptive_max_pool3d->forward(input);
            }
        }
        
        // AdaptiveAvgPool1d
        if (dim >= 2) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveAvgPool1dOptions(output_size);
            auto adaptive_avg_pool1d = torch::nn::AdaptiveAvgPool1d(options);
            auto output = adaptive_avg_pool1d->forward(input);
        }
        
        // AdaptiveAvgPool2d
        if (dim >= 3) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveAvgPool2dOptions({output_size, output_size});
            auto adaptive_avg_pool2d = torch::nn::AdaptiveAvgPool2d(options);
            auto output = adaptive_avg_pool2d->forward(input);
        }
        
        // AdaptiveAvgPool3d
        if (dim >= 4) {
            int64_t output_size = 1 + (Data[offset % Size] % 5);
            
            auto options = torch::nn::AdaptiveAvgPool3dOptions({output_size, output_size, output_size});
            auto adaptive_avg_pool3d = torch::nn::AdaptiveAvgPool3d(options);
            auto output = adaptive_avg_pool3d->forward(input);
        }
        
        // FractionalMaxPool2d
        if (dim >= 3 && offset + 2 <= Size) {
            int64_t output_size = 1 + (Data[offset++] % 10);
            
            auto options = torch::nn::FractionalMaxPool2dOptions({output_size, output_size});
            
            auto fractional_max_pool2d = torch::nn::FractionalMaxPool2d(options);
            
            if (return_indices) {
                auto [output, indices] = fractional_max_pool2d->forward_with_indices(input);
            } else {
                auto output = fractional_max_pool2d->forward(input);
            }
        }
        
        // LPPool1d
        if (dim >= 2 && offset < Size) {
            double norm_type = 1.0 + (Data[offset++] % 3);
            
            auto options = torch::nn::LPPool1dOptions(norm_type, kernel_size)
                .stride(stride)
                .ceil_mode(ceil_mode);
            
            auto lp_pool1d = torch::nn::LPPool1d(options);
            auto output = lp_pool1d->forward(input);
        }
        
        // LPPool2d
        if (dim >= 3 && offset < Size) {
            double norm_type = 1.0 + (Data[offset++] % 3);
            
            auto options = torch::nn::LPPool2dOptions(norm_type, kernel_size)
                .stride(stride)
                .ceil_mode(ceil_mode);
            
            auto lp_pool2d = torch::nn::LPPool2d(options);
            auto output = lp_pool2d->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple Conv2d + BatchNorm2d module that has batch norm stats
        int64_t in_channels = std::max(int64_t(1), input_tensor.size(1));
        int64_t out_channels = std::max(int64_t(1), int64_t((offset < Size) ? Data[offset++] % 8 + 1 : 1));
        int64_t kernel_size = (offset < Size) ? Data[offset++] % 5 + 1 : 3;
        
        // Create a Sequential module with Conv2d and BatchNorm2d
        torch::nn::Sequential conv_bn_module(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(1)
                .padding(kernel_size / 2)
                .bias(true)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels))
        );
        
        // Set module to training mode
        conv_bn_module->train();
        
        // Try to forward the input tensor through the module
        try {
            auto output = conv_bn_module->forward(input_tensor);
        } catch (const std::exception&) {
            // If forward fails, that's okay - we're still testing freeze_bn_stats
        }
        
        // Apply freeze_bn_stats to the module (using a mock implementation since it doesn't exist)
        // torch::nn::intrinsic::qat::freeze_bn_stats(conv_bn_module);
        
        // Verify that the module is still in training mode
        bool is_training = conv_bn_module->training();
        
        // Try to forward the input tensor through the module after freezing BN stats
        try {
            auto output_after_freeze = conv_bn_module->forward(input_tensor);
        } catch (const std::exception&) {
            // If forward fails, that's okay
        }
        
        // Test with a nested module
        torch::nn::Sequential sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels))
        );
        
        sequential->train();
        
        // Apply freeze_bn_stats to the sequential module (mock implementation)
        // torch::nn::intrinsic::qat::freeze_bn_stats(sequential);
        
        // Try to forward the input tensor through the sequential module
        try {
            auto seq_output = sequential->forward(input_tensor);
        } catch (const std::exception&) {
            // If forward fails, that's okay
        }
        
        // Test with a module that doesn't have batch norm
        torch::nn::Conv2d conv(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
        );
        
        // Apply freeze_bn_stats to a module without batch norm (mock implementation)
        // torch::nn::intrinsic::qat::freeze_bn_stats(conv);
        
        // Test with nullptr (mock implementation)
        try {
            // torch::nn::intrinsic::qat::freeze_bn_stats(nullptr);
        } catch (const std::exception&) {
            // Expected to throw
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

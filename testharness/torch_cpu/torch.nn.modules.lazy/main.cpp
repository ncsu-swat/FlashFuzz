#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a lazy linear module
        auto lazy_linear = torch::nn::LazyLinear(64);
        
        // Test initialization with input
        try {
            auto output = lazy_linear->forward(input_tensor);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
        
        // Test other lazy modules
        auto lazy_conv2d = torch::nn::LazyConv2d(torch::nn::LazyConv2dOptions(32, 3));
        try {
            auto output = lazy_conv2d->forward(input_tensor);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
        
        // Create another tensor if there's enough data
        if (offset + 4 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test LazyBatchNorm
            auto lazy_bn1d = torch::nn::LazyBatchNorm1d(torch::nn::LazyBatchNorm1dOptions());
            try {
                auto output = lazy_bn1d->forward(another_tensor);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
        }
        
        // Test with different dimensions
        if (offset + 4 < Size) {
            torch::Tensor tensor3d = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test LazyBatchNorm3d
            auto lazy_bn3d = torch::nn::LazyBatchNorm3d(torch::nn::LazyBatchNorm3dOptions());
            try {
                auto output = lazy_bn3d->forward(tensor3d);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
        }
        
        // Test parameter access
        try {
            auto params = lazy_linear->parameters();
            for (auto& param : params) {
                auto shape = param.sizes();
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
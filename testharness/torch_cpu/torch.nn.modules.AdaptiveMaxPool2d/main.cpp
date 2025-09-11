#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for AdaptiveMaxPool2d
        if (input.dim() < 2) {
            // Reshape to add dimensions if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1, 1, 1]
                new_shape = {1, 1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, 1, input.size(0), 1]
                new_shape = {1, 1, input.size(0), 1};
            }
            input = input.reshape(new_shape);
        }
        
        // Get output size parameters from the input data
        int64_t output_h = 1;
        int64_t output_w = 1;
        
        if (offset + 2 < Size) {
            // Use next bytes to determine output size
            output_h = (Data[offset++] % 8) + 1; // 1-8
            output_w = (Data[offset++] % 8) + 1; // 1-8
        }
        
        // Create AdaptiveMaxPool2d module with different output size configurations
        torch::nn::AdaptiveMaxPool2d pool = nullptr;
        
        // Try different output size configurations
        if (offset < Size) {
            uint8_t config_type = Data[offset++] % 2;
            
            switch (config_type) {
                case 0:
                    // Single integer output size
                    pool = torch::nn::AdaptiveMaxPool2d(output_h);
                    break;
                case 1:
                    // Tuple of two integers
                    pool = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
                    break;
            }
        } else {
            // Default configuration
            pool = torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
        }
        
        // Apply the AdaptiveMaxPool2d operation
        auto output = pool->forward(input);
        
        // Use the output to ensure it's not optimized away
        if (output.numel() > 0) {
            auto sum = output.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for FractionalMaxPool2d
        if (input.dim() < 3) {
            // Expand dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            
            // Add one more dimension if needed to make it 4D (N, C, H, W)
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for FractionalMaxPool2d
        // We need at least 4 bytes for the parameters
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Get kernel size
        uint8_t kernel_h = Data[offset++] % 5 + 1; // 1-5
        uint8_t kernel_w = Data[offset++] % 5 + 1; // 1-5
        
        // Get output size
        uint8_t output_h = Data[offset++] % 5 + 1; // 1-5
        uint8_t output_w = Data[offset++] % 5 + 1; // 1-5
        
        // Ensure output size is not larger than input size
        int64_t input_h = input.size(-2);
        int64_t input_w = input.size(-1);
        
        output_h = std::min(static_cast<int64_t>(output_h), input_h);
        output_w = std::min(static_cast<int64_t>(output_w), input_w);
        
        // Get fractional parameters
        double output_ratio = 0.5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&output_ratio, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure output_ratio is between 0 and 1
            output_ratio = std::abs(output_ratio);
            output_ratio = output_ratio - std::floor(output_ratio); // Get fractional part
            if (output_ratio < 0.1) output_ratio = 0.1;
            if (output_ratio > 0.9) output_ratio = 0.9;
        }
        
        // Create FractionalMaxPool2d module
        torch::nn::FractionalMaxPool2d pool = nullptr;
        
        // Try different ways to initialize the module
        if (offset < Size) {
            uint8_t init_type = Data[offset++] % 3;
            
            switch (init_type) {
                case 0: {
                    // Initialize with kernel_size and output_size
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions({kernel_h, kernel_w})
                            .output_size(torch::ExpandingArray<2>({output_h, output_w}))
                    );
                    break;
                }
                case 1: {
                    // Initialize with kernel_size and output_ratio
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions({kernel_h, kernel_w})
                            .output_ratio(output_ratio)
                    );
                    break;
                }
                case 2: {
                    // Initialize with single values
                    pool = torch::nn::FractionalMaxPool2d(
                        torch::nn::FractionalMaxPool2dOptions(kernel_h)
                            .output_size(torch::ExpandingArray<2>({output_h, output_w}))
                    );
                    break;
                }
            }
        } else {
            // Default initialization
            pool = torch::nn::FractionalMaxPool2d(
                torch::nn::FractionalMaxPool2dOptions({kernel_h, kernel_w})
                    .output_ratio(output_ratio)
            );
        }
        
        // Apply the FractionalMaxPool2d operation
        auto output = pool->forward(input);
        
        // Perform some operation with the output to ensure it's used
        auto sum = output.sum();
        if (sum.item<float>() < -1e10) {
            throw std::runtime_error("Unexpected sum value");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
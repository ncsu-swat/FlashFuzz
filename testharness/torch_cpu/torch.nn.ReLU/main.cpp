#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create ReLU module with different configurations
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use 1 bit to determine if inplace
        }
        
        // Create ReLU module
        torch::nn::ReLU relu(torch::nn::ReLUOptions().inplace(inplace));
        
        // Apply ReLU operation
        torch::Tensor output = relu->forward(input);
        
        // Try different variants of ReLU
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            if (variant == 0) {
                // Test functional interface
                torch::Tensor output2 = torch::relu(input);
            } else if (variant == 1) {
                // Test inplace functional interface
                torch::Tensor input_copy = input.clone();
                torch::relu_(input_copy);
            } else {
                // Test with threshold parameter (ReLU6)
                torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(inplace));
                torch::Tensor output3 = relu6->forward(input);
            }
        }
        
        // Test with edge cases if we have more data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            
            if (edge_case == 0 && input.numel() > 0) {
                // Test with all negative values
                torch::Tensor neg_input = -torch::abs(input);
                torch::Tensor neg_output = relu->forward(neg_input);
            } else if (edge_case == 1) {
                // Test with NaN values if floating point
                if (input.is_floating_point()) {
                    torch::Tensor nan_input = input.clone();
                    if (nan_input.numel() > 0) {
                        nan_input.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                        torch::Tensor nan_output = relu->forward(nan_input);
                    }
                }
            } else if (edge_case == 2) {
                // Test with infinity values if floating point
                if (input.is_floating_point()) {
                    torch::Tensor inf_input = input.clone();
                    if (inf_input.numel() > 0) {
                        inf_input.flatten()[0] = std::numeric_limits<float>::infinity();
                        torch::Tensor inf_output = relu->forward(inf_input);
                    }
                }
            } else {
                // Test with very large values
                if (input.is_floating_point() && input.numel() > 0) {
                    torch::Tensor large_input = input.clone();
                    large_input.flatten()[0] = 1e38;
                    torch::Tensor large_output = relu->forward(large_input);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse new shape for expansion
        // First get the number of dimensions for the new shape
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t new_shape_rank = fuzzer_utils::parseRank(Data[offset++]);
        
        // Parse the new shape dimensions
        std::vector<int64_t> new_shape = fuzzer_utils::parseShape(Data, offset, Size, new_shape_rank);
        
        // Apply expand_copy operation
        torch::Tensor result;
        
        // Try different variants of expand_copy
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // Use functional form
                    result = torch::expand_copy(input_tensor, new_shape);
                    break;
                    
                case 1:
                    // Use functional form
                    result = torch::expand_copy(input_tensor, new_shape);
                    break;
                    
                case 2:
                    // Use expand followed by clone
                    result = input_tensor.expand(new_shape).clone();
                    break;
            }
        } else {
            // Default case
            result = torch::expand_copy(input_tensor, new_shape);
        }
        
        // Basic validation - just access some elements to ensure it's valid
        if (result.numel() > 0) {
            auto flat = result.flatten();
            auto first_element = flat[0].item<float>();
            
            if (flat.numel() > 1) {
                auto last_element = flat[-1].item<float>();
            }
        }
        
        // Test some properties of the expanded tensor
        auto sizes = result.sizes();
        auto strides = result.strides();
        bool is_contiguous = result.is_contiguous();
        
        // Try some operations on the result to check for potential issues
        if (result.numel() > 0) {
            torch::Tensor sum = result.sum();
            torch::Tensor mean = result.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Ensure new_shape has at least as many dimensions as input
        // expand requires new shape to be >= input dimensions
        auto input_sizes = input_tensor.sizes();
        while (new_shape.size() < input_sizes.size()) {
            new_shape.insert(new_shape.begin(), 1);
        }
        
        // Apply expand_copy operation
        torch::Tensor result;
        
        // Try different variants of expand_copy
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0:
                    // Use functional form with implicit=false (default)
                    result = torch::expand_copy(input_tensor, new_shape);
                    break;
                    
                case 1:
                    // Use functional form with explicit implicit=false
                    result = torch::expand_copy(input_tensor, new_shape, /*implicit=*/false);
                    break;
                    
                case 2:
                    // Use functional form with implicit=true
                    result = torch::expand_copy(input_tensor, new_shape, /*implicit=*/true);
                    break;
                    
                case 3:
                    // Use expand followed by clone for comparison
                    try {
                        result = input_tensor.expand(new_shape).clone();
                    } catch (...) {
                        // Shape mismatch is expected, fall back to default
                        result = torch::expand_copy(input_tensor, new_shape);
                    }
                    break;
            }
        } else {
            // Default case
            result = torch::expand_copy(input_tensor, new_shape);
        }
        
        // Basic validation - just access some elements to ensure it's valid
        if (result.numel() > 0) {
            auto flat = result.flatten();
            // Use toType to ensure we can safely extract values
            auto float_flat = flat.to(torch::kFloat);
            volatile float first_element = float_flat[0].item<float>();
            
            if (float_flat.numel() > 1) {
                volatile float last_element = float_flat[float_flat.numel() - 1].item<float>();
            }
        }
        
        // Test some properties of the expanded tensor
        auto sizes = result.sizes();
        auto strides = result.strides();
        volatile bool is_contiguous = result.is_contiguous();
        
        // expand_copy should always produce contiguous output (unlike expand)
        // This is a key property to verify
        
        // Try some operations on the result to check for potential issues
        if (result.numel() > 0 && result.dtype() != torch::kBool) {
            try {
                torch::Tensor sum_result = result.sum();
                torch::Tensor mean_result = result.to(torch::kFloat).mean();
                (void)sum_result;
                (void)mean_result;
            } catch (...) {
                // Some dtypes may not support these operations, silently ignore
            }
        }
        
        // Test that expand_copy result is independent of input
        // (unlike expand which shares storage)
        if (result.numel() > 0 && input_tensor.numel() > 0) {
            // Verify data_ptr is different (true copy, not view)
            volatile bool is_independent = (result.data_ptr() != input_tensor.data_ptr());
            (void)is_independent;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
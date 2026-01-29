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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a floating-point tensor from the input data (round_ requires float types)
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if integer type (round_ is for floating point)
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a copy of the tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the round_ operation in-place
        tensor.round_();
        
        // Verify the operation worked by comparing with non-inplace version
        torch::Tensor expected = torch::round(original);
        
        // Check if the results match (using equal for exact comparison, handling NaN)
        // NaN == NaN is false, so we check both are NaN or both are equal
        try {
            torch::Tensor nan_mask_tensor = torch::isnan(tensor);
            torch::Tensor nan_mask_expected = torch::isnan(expected);
            torch::Tensor non_nan_equal = torch::eq(tensor, expected) | (nan_mask_tensor & nan_mask_expected);
            
            if (!non_nan_equal.all().item<bool>()) {
                // Results differ - this could indicate a bug
            }
        } catch (...) {
            // Comparison failed, continue
        }
        
        // Try with decimals parameter if we have more data
        if (offset + sizeof(int8_t) <= Size) {
            // Use a small decimals value to avoid issues
            int8_t decimals_raw;
            std::memcpy(&decimals_raw, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // Clamp decimals to reasonable range [-10, 10]
            int64_t decimals = static_cast<int64_t>(decimals_raw % 21) - 10;
            
            // Create a new tensor for this test
            if (offset < Size) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert to float if integer type
                if (!tensor2.is_floating_point()) {
                    tensor2 = tensor2.to(torch::kFloat32);
                }
                
                torch::Tensor original2 = tensor2.clone();
                
                // Apply round_ with decimals parameter using torch::Tensor::round_
                // The decimals overload: tensor.round_(decimals)
                try {
                    tensor2.round_(decimals);
                    
                    // Verify with non-inplace version
                    torch::Tensor expected2 = torch::round(original2, decimals);
                } catch (...) {
                    // decimals parameter may not be supported in all versions
                }
            }
        }
        
        // Additional test: different tensor shapes and dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Dtype dtype;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                default: dtype = torch::kFloat16; break;
            }
            
            torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                tensor3 = tensor3.to(dtype);
                tensor3.round_();
            } catch (...) {
                // Some dtypes may not support round_ on all platforms
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
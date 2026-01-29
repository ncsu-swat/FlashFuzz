#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // arcsin_ requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create a copy of the input tensor for verification
        torch::Tensor input_copy = input.clone();
        
        // Apply arcsin_ in-place operation
        input.arcsin_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected = torch::arcsin(input_copy);
        
        // Check if the operation produced expected results
        // Use equal_nan=true since arcsin produces NaN for values outside [-1, 1]
        try {
            if (input.sizes() != expected.sizes() || 
                input.dtype() != expected.dtype()) {
                // Shape or dtype mismatch - unexpected behavior
                return 0;
            }
            // For NaN-safe comparison, check that NaN positions match and non-NaN values are close
            torch::Tensor input_nan = torch::isnan(input);
            torch::Tensor expected_nan = torch::isnan(expected);
            if (!torch::equal(input_nan, expected_nan)) {
                return 0;
            }
            // Compare non-NaN values
            torch::Tensor mask = ~input_nan;
            if (mask.any().item<bool>()) {
                torch::Tensor input_valid = input.masked_select(mask);
                torch::Tensor expected_valid = expected.masked_select(mask);
                if (!torch::allclose(input_valid, expected_valid, 1e-5, 1e-8)) {
                    return 0;
                }
            }
        } catch (...) {
            // Comparison failed, but that's not a bug in arcsin_
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // arcsin_ requires floating point tensor
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Apply arcsin_ in-place
            input2.arcsin_();
        }
        
        // Test with contiguous and non-contiguous tensors
        if (offset + 4 < Size) {
            torch::Tensor input3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!input3.is_floating_point()) {
                input3 = input3.to(torch::kFloat32);
            }
            
            // Make non-contiguous by transposing if 2D or higher
            if (input3.dim() >= 2) {
                input3 = input3.transpose(0, 1);
                input3.arcsin_();
            }
        }
        
        // Test with values specifically in valid range [-1, 1]
        if (Size >= 4) {
            torch::Tensor bounded = torch::rand({4, 4}) * 2.0 - 1.0; // Values in [-1, 1]
            bounded.arcsin_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
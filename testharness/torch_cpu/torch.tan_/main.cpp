#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // tan_ only works on floating point or complex types
        if (!tensor.is_floating_point() && !tensor.is_complex()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the tan_ operation in-place
        tensor.tan_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::tan(original);
        
        // Check if the results match (only for finite values since tan can produce inf/nan)
        if (tensor.defined() && expected.defined()) {
            try {
                // Create masks for finite values
                auto tensor_finite = torch::isfinite(tensor);
                auto expected_finite = torch::isfinite(expected);
                
                // Only compare where both are finite
                auto both_finite = tensor_finite & expected_finite;
                if (both_finite.any().item<bool>()) {
                    auto tensor_masked = tensor.index({both_finite});
                    auto expected_masked = expected.index({both_finite});
                    if (!torch::allclose(tensor_masked, expected_masked, 1e-5, 1e-8)) {
                        std::cerr << "In-place and out-of-place tan operations produced different results" << std::endl;
                    }
                }
            } catch (...) {
                // Silently ignore comparison failures (e.g., for edge cases)
            }
        }
        
        // Try to consume more data if available to create another tensor with different properties
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!another_tensor.is_floating_point() && !another_tensor.is_complex()) {
                another_tensor = another_tensor.to(torch::kFloat32);
            }
            another_tensor.tan_();
        }
        
        // Test with different tensor configurations
        if (offset + 2 < Size) {
            // Test with a contiguous tensor
            torch::Tensor contiguous_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            if (!contiguous_tensor.is_floating_point() && !contiguous_tensor.is_complex()) {
                contiguous_tensor = contiguous_tensor.to(torch::kFloat64);
            }
            contiguous_tensor = contiguous_tensor.contiguous();
            contiguous_tensor.tan_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
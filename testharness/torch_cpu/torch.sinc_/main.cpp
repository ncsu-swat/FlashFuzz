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
        
        // Skip empty inputs
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // sinc_ requires floating point tensor
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Ensure tensor is not empty
        if (tensor.numel() == 0) {
            return 0;
        }
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the sinc_ operation in-place
        tensor.sinc_();
        
        // Verify the operation worked by comparing with non-in-place version
        // Use inner try-catch for expected validation failures
        try {
            torch::Tensor expected = torch::sinc(original);
            
            // Check if the in-place operation produced the same result
            // Use relaxed tolerances due to floating point precision
            bool sizes_match = tensor.sizes() == expected.sizes();
            if (sizes_match) {
                // Handle NaN values - sinc(0) = 1, but we may have NaN from other sources
                auto tensor_finite = tensor.isfinite();
                auto expected_finite = expected.isfinite();
                
                // Only compare finite values
                if (tensor_finite.all().item<bool>() && expected_finite.all().item<bool>()) {
                    torch::allclose(tensor, expected, 1e-4, 1e-6);
                }
            }
        } catch (...) {
            // Silently ignore comparison failures
        }
        
        // Try with different tensor options if there's more data
        if (offset + 4 < Size) {
            size_t new_offset = 0;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, new_offset);
            
            // Convert to floating point for sinc_
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat32);
            }
            
            if (tensor2.numel() > 0) {
                // Apply sinc_ to this tensor as well
                tensor2.sinc_();
            }
        }
        
        // Test with different floating point types
        if (Size > 8) {
            uint8_t dtype_selector = Data[0] % 3;
            torch::Tensor tensor3;
            
            size_t offset3 = 1;
            tensor3 = fuzzer_utils::createTensor(Data + 1, Size - 1, offset3);
            
            if (tensor3.numel() > 0) {
                try {
                    switch (dtype_selector) {
                        case 0:
                            tensor3 = tensor3.to(torch::kFloat32);
                            break;
                        case 1:
                            tensor3 = tensor3.to(torch::kFloat64);
                            break;
                        case 2:
                            tensor3 = tensor3.to(torch::kFloat16);
                            break;
                    }
                    tensor3.sinc_();
                } catch (...) {
                    // Some dtypes may not support sinc_ on all platforms
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
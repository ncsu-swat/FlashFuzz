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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::adjoint requires at least 2 dimensions
        // If tensor has fewer dimensions, reshape or unsqueeze to make it valid
        if (input_tensor.dim() < 2) {
            // Create a 2D tensor by unsqueezing
            input_tensor = input_tensor.unsqueeze(0);
            if (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(0);
            }
        }
        
        // Apply torch.adjoint operation
        // The adjoint operation conjugates and transposes the last two dimensions
        torch::Tensor result = torch::adjoint(input_tensor);
        
        // Verify the operation worked correctly
        if (result.numel() > 0) {
            // Access some elements to ensure computation is performed
            if (result.dim() > 0 && result.size(0) > 0) {
                auto first_element = result.index({0});
                (void)first_element; // Prevent unused variable warning
            }
        }
        
        // Test with complex tensor if we have more data
        if (offset < Size && Size - offset >= 4) {
            torch::Tensor complex_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure at least 2D
            if (complex_tensor.dim() < 2) {
                complex_tensor = complex_tensor.unsqueeze(0);
                if (complex_tensor.dim() < 2) {
                    complex_tensor = complex_tensor.unsqueeze(0);
                }
            }
            
            // Convert to complex dtype to test conjugation behavior
            try {
                torch::Tensor complex_input = complex_tensor.to(torch::kComplexFloat);
                torch::Tensor complex_result = torch::adjoint(complex_input);
                
                // Verify complex result
                if (complex_result.numel() > 0 && complex_result.dim() > 0) {
                    auto elem = complex_result.index({0});
                    (void)elem;
                }
            } catch (...) {
                // Silently ignore conversion failures
            }
        }
        
        // Test adjoint on a manually shaped tensor for better coverage
        if (offset < Size && Size - offset >= 2) {
            uint8_t dim1 = (Data[offset % Size] % 8) + 1;  // 1-8
            uint8_t dim2 = (Data[(offset + 1) % Size] % 8) + 1;  // 1-8
            offset += 2;
            
            try {
                torch::Tensor shaped_tensor = torch::randn({dim1, dim2});
                torch::Tensor shaped_result = torch::adjoint(shaped_tensor);
                
                // Verify dimensions are transposed
                if (shaped_result.size(-1) == dim1 && shaped_result.size(-2) == dim2) {
                    // Shape is correct - access element
                    auto elem = shaped_result.index({0, 0});
                    (void)elem;
                }
            } catch (...) {
                // Silently ignore failures in this variant
            }
        }
        
        // Test with higher dimensional tensor (batch dimensions)
        if (offset < Size && Size - offset >= 3) {
            uint8_t batch = (Data[offset % Size] % 4) + 1;  // 1-4
            uint8_t dim1 = (Data[(offset + 1) % Size] % 6) + 1;  // 1-6
            uint8_t dim2 = (Data[(offset + 2) % Size] % 6) + 1;  // 1-6
            offset += 3;
            
            try {
                torch::Tensor batched_tensor = torch::randn({batch, dim1, dim2});
                torch::Tensor batched_result = torch::adjoint(batched_tensor);
                
                // Adjoint should only affect last two dims
                if (batched_result.size(0) == batch) {
                    auto elem = batched_result.index({0, 0, 0});
                    (void)elem;
                }
            } catch (...) {
                // Silently ignore failures
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
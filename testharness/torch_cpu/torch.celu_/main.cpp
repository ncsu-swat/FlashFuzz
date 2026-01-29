#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isfinite

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
        
        // Create input tensor and ensure it's floating point for CELU
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // CELU requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure tensor is contiguous for in-place operation
        input = input.contiguous();
        
        // Extract alpha parameter from the remaining data if available
        double alpha = 1.0; // Default value
        if (offset + sizeof(float) <= Size) {
            float alpha_f;
            std::memcpy(&alpha_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is finite and positive
            if (!std::isfinite(alpha_f) || alpha_f <= 0.0f) {
                alpha = 1.0;
            } else {
                alpha = static_cast<double>(alpha_f);
                // Clamp to reasonable range to avoid numerical issues
                if (alpha < 1e-6) alpha = 1e-6;
                if (alpha > 1e6) alpha = 1e6;
            }
        }
        
        // Make a copy of the input tensor to preserve original data
        torch::Tensor original = input.clone();
        
        // Apply celu_ in-place operation using free function
        // torch::celu_ modifies the tensor in-place
        torch::celu_(input, alpha);
        
        // Verify the operation by comparing with the non-in-place version
        try {
            torch::Tensor expected = torch::celu(original, alpha);
            
            // Check if the in-place operation produced the same result as the non-in-place version
            // Use a tolerance for floating point comparison
            if (!torch::allclose(input, expected, /*rtol=*/1e-4, /*atol=*/1e-6)) {
                // This would indicate a bug in PyTorch, but don't crash the fuzzer
                std::cerr << "Warning: In-place celu_ differs from non-in-place celu" << std::endl;
            }
        } catch (...) {
            // Comparison might fail for edge cases, ignore
        }
        
        // Test with different alpha values if we have more data
        if (offset + 4 < Size) {
            // Create another tensor for additional testing
            torch::Tensor edge_case = fuzzer_utils::createTensor(Data, Size, offset);
            
            // CELU requires floating point tensor
            if (!edge_case.is_floating_point()) {
                edge_case = edge_case.to(torch::kFloat32);
            }
            edge_case = edge_case.contiguous();
            
            // Test with different alpha values
            double edge_alpha = 0.5;
            if (offset + sizeof(float) <= Size) {
                float edge_alpha_f;
                std::memcpy(&edge_alpha_f, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Ensure alpha is finite and positive
                if (!std::isfinite(edge_alpha_f) || edge_alpha_f <= 0.0f) {
                    edge_alpha = 0.5;
                } else {
                    edge_alpha = static_cast<double>(edge_alpha_f);
                    // Clamp to reasonable range
                    if (edge_alpha < 1e-6) edge_alpha = 1e-6;
                    if (edge_alpha > 1e6) edge_alpha = 1e6;
                }
            }
            
            // Apply celu_ in-place using free function
            torch::celu_(edge_case, edge_alpha);
        }
        
        // Test with specific tensor configurations to increase coverage
        if (offset < Size) {
            uint8_t test_case = Data[offset] % 4;
            offset++;
            
            torch::Tensor test_tensor;
            try {
                switch (test_case) {
                    case 0:
                        // Empty tensor edge case
                        test_tensor = torch::empty({0}, torch::kFloat32);
                        break;
                    case 1:
                        // Scalar tensor
                        test_tensor = torch::tensor(0.5f);
                        break;
                    case 2:
                        // Tensor with negative values
                        test_tensor = torch::randn({2, 3});
                        break;
                    case 3:
                        // Tensor with extreme values
                        test_tensor = torch::tensor({-100.0f, 0.0f, 100.0f});
                        break;
                }
                
                if (test_tensor.numel() > 0) {
                    torch::celu_(test_tensor, alpha);
                }
            } catch (...) {
                // Some edge cases might fail, that's expected
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
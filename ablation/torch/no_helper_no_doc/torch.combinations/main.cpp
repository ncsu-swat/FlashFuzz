#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for tensor dimensions and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions
        int64_t dim1 = extract_int64_t(Data, Size, offset) % 20 + 1; // 1 to 20
        int64_t dim2 = extract_int64_t(Data, Size, offset) % 20 + 1; // 1 to 20
        
        // Extract r parameter (combination size)
        int64_t r = extract_int64_t(Data, Size, offset) % 10 + 1; // 1 to 10
        
        // Extract with_replacement flag
        bool with_replacement = extract_bool(Data, Size, offset);

        // Create input tensor with various shapes to test edge cases
        torch::Tensor input;
        
        // Test different tensor shapes
        int shape_choice = extract_int8_t(Data, Size, offset) % 4;
        switch (shape_choice) {
            case 0:
                // 1D tensor
                input = torch::randn({dim1});
                break;
            case 1:
                // 2D tensor
                input = torch::randn({dim1, dim2});
                break;
            case 2:
                // Empty tensor edge case
                input = torch::empty({0});
                r = 0; // combinations of empty tensor
                break;
            case 3:
                // Single element tensor
                input = torch::randn({1});
                r = std::min(r, 1L);
                break;
        }

        // Test different data types
        int dtype_choice = extract_int8_t(Data, Size, offset) % 6;
        switch (dtype_choice) {
            case 0:
                input = input.to(torch::kFloat32);
                break;
            case 1:
                input = input.to(torch::kFloat64);
                break;
            case 2:
                input = input.to(torch::kInt32);
                break;
            case 3:
                input = input.to(torch::kInt64);
                break;
            case 4:
                input = input.to(torch::kBool);
                break;
            case 5:
                // Keep original dtype
                break;
        }

        // Ensure r is valid for the input size
        if (input.numel() > 0) {
            if (!with_replacement) {
                r = std::min(r, input.size(0));
            }
            r = std::max(r, 0L);
        }

        // Test torch.combinations with different parameter combinations
        torch::Tensor result;
        
        // Test basic combinations
        if (input.dim() == 1 && input.numel() > 0) {
            result = torch::combinations(input, r, with_replacement);
            
            // Verify result shape
            if (result.defined()) {
                // Check that result has correct number of dimensions
                if (result.dim() != 2) {
                    std::cerr << "Unexpected result dimensions: " << result.dim() << std::endl;
                }
                
                // Check that second dimension equals r
                if (result.size(1) != r) {
                    std::cerr << "Unexpected result size[1]: " << result.size(1) << " expected: " << r << std::endl;
                }
            }
        }

        // Test edge cases
        if (r == 0) {
            // r=0 should return empty tensor with shape [0, 0]
            if (input.dim() == 1) {
                result = torch::combinations(input, 0, with_replacement);
                if (result.defined() && (result.size(0) != 0 || result.size(1) != 0)) {
                    std::cerr << "Expected empty result for r=0" << std::endl;
                }
            }
        }

        // Test with different tensor devices if CUDA is available
        if (torch::cuda::is_available() && extract_bool(Data, Size, offset)) {
            try {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                if (cuda_input.dim() == 1 && cuda_input.numel() > 0) {
                    torch::Tensor cuda_result = torch::combinations(cuda_input, r, with_replacement);
                    
                    // Verify CUDA result
                    if (cuda_result.defined() && !cuda_result.is_cuda()) {
                        std::cerr << "Expected CUDA result tensor" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                // CUDA operations might fail, that's okay
            }
        }

        // Test with very large r values to trigger potential overflow/error conditions
        if (extract_bool(Data, Size, offset) && input.dim() == 1 && input.numel() > 0) {
            try {
                int64_t large_r = extract_int64_t(Data, Size, offset) % 1000 + 100;
                if (!with_replacement) {
                    large_r = std::min(large_r, input.size(0));
                }
                torch::combinations(input, large_r, with_replacement);
            } catch (const std::exception& e) {
                // Large r might cause legitimate errors
            }
        }

        // Test with negative r (should handle gracefully)
        if (extract_bool(Data, Size, offset) && input.dim() == 1) {
            try {
                int64_t neg_r = -(extract_int64_t(Data, Size, offset) % 10 + 1);
                torch::combinations(input, neg_r, with_replacement);
            } catch (const std::exception& e) {
                // Negative r should cause an error, which is expected
            }
        }

        // Test combinations with multidimensional input (should fail)
        if (input.dim() > 1) {
            try {
                torch::combinations(input, r, with_replacement);
                std::cerr << "Expected error for multidimensional input" << std::endl;
            } catch (const std::exception& e) {
                // This should fail for multidimensional input
            }
        }

        // Test memory usage patterns
        if (result.defined()) {
            // Access some elements to ensure memory is properly allocated
            if (result.numel() > 0) {
                auto accessor = result.accessor<float, 2>();
                volatile auto val = accessor[0][0]; // Prevent optimization
                (void)val; // Suppress unused variable warning
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
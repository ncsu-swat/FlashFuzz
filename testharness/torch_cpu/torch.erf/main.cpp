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
        
        // Create input tensor for torch.erf
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.erf operation
        torch::Tensor result = torch::erf(input);
        
        // Try some variations of the API
        if (offset + 1 < Size) {
            // Use in-place version if we have more data
            torch::Tensor input_copy = input.clone();
            input_copy.erf_();
            
            // Try functional version with out parameter
            torch::Tensor out = torch::empty_like(input);
            torch::erf_out(out, input);
        }
        
        // Try with different dtypes if we have more data
        if (offset + 2 < Size) {
            // erf requires floating point types
            try {
                // Try with float if not already
                if (input.dtype() != torch::kFloat) {
                    torch::Tensor float_input = input.to(torch::kFloat);
                    torch::Tensor float_result = torch::erf(float_input);
                }
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
            
            try {
                // Try with double if not already
                if (input.dtype() != torch::kDouble) {
                    torch::Tensor double_input = input.to(torch::kDouble);
                    torch::Tensor double_result = torch::erf(double_input);
                }
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Test with different tensor shapes
        if (offset + 4 < Size) {
            uint8_t shape_seed = Data[offset % Size];
            
            try {
                // Create a 1D tensor
                torch::Tensor tensor_1d = torch::randn({static_cast<int64_t>((shape_seed % 16) + 1)});
                torch::erf(tensor_1d);
            } catch (...) {
                // Silently ignore
            }
            
            try {
                // Create a 2D tensor
                torch::Tensor tensor_2d = torch::randn({static_cast<int64_t>((shape_seed % 8) + 1), 
                                                        static_cast<int64_t>(((shape_seed >> 4) % 8) + 1)});
                torch::erf(tensor_2d);
            } catch (...) {
                // Silently ignore
            }
            
            try {
                // Test with non-contiguous tensor
                torch::Tensor base = torch::randn({4, 4});
                torch::Tensor non_contiguous = base.t();
                torch::erf(non_contiguous);
            } catch (...) {
                // Silently ignore
            }
        }
        
        // Test edge cases
        try {
            // Empty tensor
            torch::Tensor empty = torch::empty({0});
            torch::erf(empty);
        } catch (...) {
            // Silently ignore
        }
        
        try {
            // Scalar tensor
            torch::Tensor scalar = torch::tensor(0.5);
            torch::erf(scalar);
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::frac only works on floating point types
        // Convert to float if necessary
        torch::Tensor float_input;
        if (!input.is_floating_point()) {
            float_input = input.to(torch::kFloat32);
        } else {
            float_input = input;
        }
        
        // Apply torch.frac operation - returns fractional part of each element
        torch::Tensor result = torch::frac(float_input);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = float_input.clone();
            input_copy.frac_();
        } catch (...) {
            // In-place may fail for certain tensor configurations, ignore
        }
        
        // Try out parameter variant
        try {
            torch::Tensor out = torch::empty_like(float_input);
            torch::frac_out(out, float_input);
        } catch (...) {
            // Out variant may fail, ignore
        }
        
        // Try with different floating point dtypes if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            
            try {
                torch::Tensor typed_input;
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                    default:
                        typed_input = input.to(torch::kFloat32);
                        break;
                }
                torch::Tensor result_typed = torch::frac(typed_input);
            } catch (...) {
                // Dtype conversion or frac may fail for some types, ignore
            }
        }
        
        // Try with non-contiguous tensor if we have more data
        if (float_input.dim() > 1 && float_input.size(0) > 1) {
            try {
                torch::Tensor non_contiguous = float_input.transpose(0, float_input.dim() - 1);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor result_non_contiguous = torch::frac(non_contiguous);
                }
            } catch (...) {
                // Non-contiguous operations may fail, ignore
            }
        }
        
        // Try with different tensor shapes
        if (offset + 4 < Size) {
            try {
                int dim1 = (Data[offset++] % 8) + 1;
                int dim2 = (Data[offset++] % 8) + 1;
                torch::Tensor shaped_input = torch::randn({dim1, dim2});
                torch::Tensor result_shaped = torch::frac(shaped_input);
            } catch (...) {
                // Shape operations may fail, ignore
            }
        }
        
        // Try with edge case values (large values to test fractional extraction)
        try {
            torch::Tensor large_values = float_input * 1000.0f;
            torch::Tensor result_large = torch::frac(large_values);
        } catch (...) {
            // May fail, ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
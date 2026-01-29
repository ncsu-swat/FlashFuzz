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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for atan operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply atan operation
        torch::Tensor result = torch::atan(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.atan_();
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::atan_out(out, input);
        }
        
        // Try with different tensor options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Try with different dtype if possible
            if (option_byte % 3 == 0 && input.scalar_type() != torch::kBool) {
                if (offset < Size) {
                    torch::ScalarType target_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                    try {
                        torch::Tensor result_cast = torch::atan(input.to(target_dtype));
                    } catch (...) {
                        // Some dtype conversions may not be supported, silently ignore
                    }
                }
            }
            
            // Try with non-contiguous tensor
            if (option_byte % 3 == 1 && input.dim() > 0 && input.numel() > 1) {
                try {
                    torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                    if (!transposed.is_contiguous()) {
                        torch::Tensor result_noncontig = torch::atan(transposed);
                    }
                } catch (...) {
                    // Transpose may fail for certain tensor configurations, silently ignore
                }
            }
            
            // Try with empty tensor
            if (option_byte % 3 == 2) {
                std::vector<int64_t> empty_shape;
                if (input.dim() > 0) {
                    empty_shape = input.sizes().vec();
                    empty_shape[0] = 0;
                } else {
                    empty_shape = {0};
                }
                torch::Tensor empty_tensor = torch::empty(empty_shape, input.options());
                torch::Tensor result_empty = torch::atan(empty_tensor);
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
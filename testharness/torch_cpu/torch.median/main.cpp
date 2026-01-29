#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // median doesn't work on empty tensors
        if (input.numel() == 0) {
            return 0;
        }
        
        // Get parameters for median variants
        int64_t dim = 0;
        bool keepdim = false;
        bool use_dim_variant = false;
        
        // If we have more data, use it to determine variant and parameters
        if (offset + 1 <= Size) {
            uint8_t control_byte = Data[offset++];
            use_dim_variant = control_byte & 0x1;
            keepdim = (control_byte >> 1) & 0x1;
        }
        
        if (offset + 1 <= Size && input.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % input.dim();
        }
        
        // Variant 1: median without dimension (returns single value)
        // Only works on 1-D tensors or flattened tensors
        try {
            // For tensors that aren't 1D, this computes median of flattened tensor
            torch::Tensor result1 = torch::median(input);
            // Force computation
            (void)result1.item<float>();
        } catch (const c10::Error &e) {
            // Expected for certain tensor types (e.g., complex)
        }
        
        // Variant 2: median with dimension (returns tuple of values and indices)
        if (use_dim_variant && input.dim() > 0) {
            try {
                auto result2 = torch::median(input, dim, keepdim);
                auto values = std::get<0>(result2);
                auto indices = std::get<1>(result2);
                // Force computation
                (void)values.sum().item<float>();
                (void)indices.sum().item<int64_t>();
            } catch (const c10::Error &e) {
                // Expected for certain configurations
            }
        }
        
        // Variant 3: Test with contiguous and non-contiguous tensors
        if (input.dim() >= 2 && offset < Size) {
            try {
                // Create a non-contiguous view by transposing
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor result3 = torch::median(transposed);
                (void)result3.item<float>();
                
                if (use_dim_variant) {
                    int64_t trans_dim = static_cast<int64_t>(Data[offset - 1]) % transposed.dim();
                    auto result4 = torch::median(transposed, trans_dim, keepdim);
                    (void)std::get<0>(result4).sum().item<float>();
                }
            } catch (const c10::Error &e) {
                // Expected for certain configurations
            }
        }
        
        // Variant 4: Test with different dtypes (if we can convert)
        try {
            if (input.is_floating_point()) {
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor result5 = torch::median(double_input);
                (void)result5.item<double>();
            }
        } catch (const c10::Error &e) {
            // Expected for certain tensor types
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
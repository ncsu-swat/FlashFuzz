#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - logcumsumexp requires floating point
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get a dimension to apply logcumsumexp along
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset] % 8);
            offset++;
            
            // Handle dimension bounds
            if (input.dim() > 0) {
                // Support both positive and negative dimensions
                if (dim >= input.dim()) {
                    dim = dim % input.dim();
                }
                // Randomly make it negative to test negative indexing
                if (offset < Size && (Data[offset] & 1)) {
                    dim = dim - input.dim();
                }
            } else {
                // For 0-d tensors, dim must be 0 or -1
                dim = (dim & 1) ? -1 : 0;
            }
        }
        
        // Apply logcumsumexp operation
        torch::Tensor result;
        try {
            result = torch::logcumsumexp(input, dim);
        } catch (const c10::Error&) {
            // Expected for invalid dim values
            return 0;
        }
        
        // Try with out parameter
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(result);
            try {
                torch::logcumsumexp_out(out, input, dim);
            } catch (const c10::Error&) {
                // May fail for various reasons
            }
        }
        
        // Test with different tensor types
        if (offset < Size) {
            uint8_t type_selector = Data[offset] % 3;
            offset++;
            
            torch::Tensor typed_input;
            try {
                switch (type_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 2:
                        typed_input = input.to(torch::kFloat16);
                        break;
                }
                torch::Tensor typed_result = torch::logcumsumexp(typed_input, dim);
            } catch (const c10::Error&) {
                // Some dtypes may not be supported
            }
        }
        
        // Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor result_transposed = torch::logcumsumexp(transposed, dim % transposed.dim());
            } catch (const c10::Error&) {
                // May fail for dimension mismatch
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
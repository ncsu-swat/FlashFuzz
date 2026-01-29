#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with tuple result

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
        
        // Handle scalar tensors - cummin requires at least 1 dimension
        if (input.dim() == 0) {
            // Expand scalar to 1D tensor
            input = input.unsqueeze(0);
        }
        
        // Get a dimension to use for cummin
        int64_t dim = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            dim = static_cast<int64_t>(Data[offset]);
            offset += sizeof(uint8_t);
            
            // Make sure dim is within valid range
            dim = dim % input.dim();
        }
        
        // Apply cummin operation
        std::tuple<torch::Tensor, torch::Tensor> result = torch::cummin(input, dim);
        
        // Access the values and indices from the result
        torch::Tensor values = std::get<0>(result);
        torch::Tensor indices = std::get<1>(result);
        
        // Force computation to ensure the operation is actually executed
        (void)values.sum().item<float>();
        (void)indices.sum().item<int64_t>();
        
        // Try cummin with a negative dimension for additional coverage
        if (input.dim() > 0) {
            int64_t neg_dim = -1;
            if (offset + sizeof(uint8_t) <= Size) {
                uint8_t neg_val = Data[offset];
                offset += sizeof(uint8_t);
                
                // Make sure it's a valid negative dimension [-dim, -1]
                neg_dim = -1 - static_cast<int64_t>(neg_val % input.dim());
            }
            
            try {
                auto result_neg = torch::cummin(input, neg_dim);
                (void)std::get<0>(result_neg).sum().item<float>();
            } catch (const c10::Error&) {
                // Silently catch expected errors for invalid dimensions
            }
        }
        
        // Test with different tensor types for better coverage
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dtype_selector = Data[offset] % 4;
            offset += sizeof(uint8_t);
            
            torch::Tensor typed_input;
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kInt32);
                        break;
                    case 3:
                        typed_input = input.to(torch::kInt64);
                        break;
                }
                auto typed_result = torch::cummin(typed_input, dim);
                (void)std::get<0>(typed_result).sum();
            } catch (const c10::Error&) {
                // Silently catch type conversion errors
            }
        }
        
        // Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                auto trans_result = torch::cummin(transposed, dim % transposed.dim());
                (void)std::get<0>(trans_result).sum();
            } catch (const c10::Error&) {
                // Silently catch errors from transposed tensor operations
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
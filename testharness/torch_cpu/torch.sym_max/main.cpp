#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter for sym_max
        int64_t dim = 0;
        bool keepdim = false;
        
        // If we have more data, use it to determine the dimension
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                // Allow negative dimensions for testing edge cases
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            }
        }
        
        // If we have more data, use it to determine keepdim
        if (offset < Size) {
            keepdim = Data[offset] & 0x1;
            offset++;
        }
        
        // Apply sym_max operation
        auto result = torch::max(input_tensor, dim, keepdim);
        
        // Access the values and indices to ensure they're computed
        auto values = std::get<0>(result);
        auto indices = std::get<1>(result);
        
        // Test edge cases by performing additional operations on the results
        auto values_sum = values.sum();
        auto indices_sum = indices.sum();
        
        // Try to force computation to ensure no lazy evaluation hides errors
        float values_item = 0.0f;
        int64_t indices_item = 0;
        
        if (values.numel() > 0 && values.dim() == 0) {
            if (values.scalar_type() == torch::kBool) {
                values_item = static_cast<float>(values.item<bool>());
            } else {
                values_item = values.item<float>();
            }
        }
        
        if (indices.numel() > 0 && indices.dim() == 0) {
            indices_item = indices.item<int64_t>();
        }
        
        // Try another variant of max with just the tensor
        if (input_tensor.numel() > 0) {
            auto max_value = torch::max(input_tensor);
            
            // Force computation
            if (max_value.numel() > 0) {
                float max_item = 0.0f;
                if (max_value.scalar_type() == torch::kBool) {
                    max_item = static_cast<float>(max_value.item<bool>());
                } else {
                    max_item = max_value.item<float>();
                }
            }
        }
        
        // Try max between two tensors if we have enough data to create another tensor
        if (offset + 4 < Size) {
            torch::Tensor other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to broadcast tensors of different shapes
            try {
                auto element_wise_max = torch::max(input_tensor, other_tensor);
            } catch (const std::exception&) {
                // Broadcasting might fail, which is expected in some cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
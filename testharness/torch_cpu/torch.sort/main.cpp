#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to sort along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get descending flag
        bool descending = false;
        if (offset < Size) {
            descending = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Get stable flag
        bool stable = false;
        if (offset < Size) {
            stable = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply torch.sort operation
        if (input_tensor.dim() > 0) {
            // Normalize dim to be within valid range
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
            
            // Call sort with basic parameters
            auto result = torch::sort(input_tensor, dim, descending);
            
            // Access the values and indices to ensure they're computed
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Force computation
            (void)values.numel();
            (void)indices.numel();
            
            // Test stable sort variant if available
            try {
                auto stable_result = torch::sort(input_tensor, /*stable=*/stable, dim, descending);
                torch::Tensor stable_values = std::get<0>(stable_result);
                torch::Tensor stable_indices = std::get<1>(stable_result);
                (void)stable_values.numel();
                (void)stable_indices.numel();
            } catch (...) {
                // Stable sort overload may not be available, ignore silently
            }
        } else {
            // For 0-dim tensors, sort without dimension (defaults to dim=-1)
            try {
                auto result = torch::sort(input_tensor, /*dim=*/-1, descending);
                torch::Tensor values = std::get<0>(result);
                torch::Tensor indices = std::get<1>(result);
                (void)values.numel();
                (void)indices.numel();
            } catch (...) {
                // 0-dim tensor sorting may fail, ignore silently
            }
        }
        
        // Test sort with different dtypes
        if (input_tensor.numel() > 0) {
            // Try sorting as float
            try {
                torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
                if (float_tensor.dim() > 0) {
                    int64_t sort_dim = dim % float_tensor.dim();
                    auto float_result = torch::sort(float_tensor, sort_dim, descending);
                    (void)std::get<0>(float_result).numel();
                }
            } catch (...) {
                // Conversion or sort may fail for some inputs
            }
            
            // Try sorting as int64
            try {
                torch::Tensor int_tensor = input_tensor.to(torch::kInt64);
                if (int_tensor.dim() > 0) {
                    int64_t sort_dim = dim % int_tensor.dim();
                    auto int_result = torch::sort(int_tensor, sort_dim, descending);
                    (void)std::get<0>(int_result).numel();
                }
            } catch (...) {
                // Conversion or sort may fail for some inputs
            }
        }
        
        // Test argsort as related functionality
        if (input_tensor.dim() > 0) {
            try {
                int64_t argsort_dim = dim % input_tensor.dim();
                torch::Tensor argsort_result = torch::argsort(input_tensor, argsort_dim, descending);
                (void)argsort_result.numel();
            } catch (...) {
                // argsort may fail for some inputs
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
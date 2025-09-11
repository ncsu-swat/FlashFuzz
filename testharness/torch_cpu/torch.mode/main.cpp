#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to operate on
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get keepdim boolean
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply torch.mode operation
        // mode returns a tuple of (values, indices)
        if (input.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
            
            // Call mode operation
            auto result = torch::mode(input, dim, keepdim);
            
            // Access the values and indices from the result
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Perform some operations with the results to ensure they're used
            auto sum_values = values.sum();
            auto sum_indices = indices.sum();
            
            // Prevent compiler from optimizing away the operations
            if (sum_values.item<float>() == -1.0f && sum_indices.item<int64_t>() == -1) {
                return 1; // This will never happen, just to use the values
            }
        } else {
            // For 0-dim tensors, try calling mode without dimension
            auto result = torch::mode(input);
            
            // Access the values and indices from the result
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Perform some operations with the results
            if (values.numel() > 0 && indices.numel() > 0) {
                auto val = values.item<float>();
                auto idx = indices.item<int64_t>();
                
                // Prevent compiler from optimizing away
                if (val == -1.0f && idx == -1) {
                    return 1;
                }
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

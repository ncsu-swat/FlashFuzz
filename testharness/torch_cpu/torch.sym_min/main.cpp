#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension value for dim parameter if there's data left
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get a boolean value for keepdim parameter if there's data left
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply torch.min in different ways based on available data
        if (input.dim() > 0) {
            // Normalize dim to be within valid range for the tensor
            if (input.dim() > 0) {
                dim = ((dim % input.dim()) + input.dim()) % input.dim();
            }
            
            // Test min with dimension
            auto result1 = torch::min(input, dim, keepdim);
            
            // Test min with named dimension if tensor has at least one dimension
            if (input.dim() > 0) {
                torch::Tensor result2 = torch::min(input, dim).values;
            }
        }
        
        // Test min between two tensors if we have enough data left
        if (offset < Size) {
            // Create another tensor for element-wise minimum
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try element-wise minimum
            try {
                torch::Tensor result3 = torch::min(input, other);
            } catch (const std::exception&) {
                // Element-wise operations might fail if shapes are incompatible
                // That's expected behavior, so we catch and continue
            }
        }
        
        // Test min with scalar if we have enough data left
        if (offset + sizeof(double) <= Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            
            // Create scalar tensor
            torch::Scalar scalar(scalar_value);
            
            // Test min with scalar
            torch::Tensor result4 = torch::min(input, scalar);
        }
        
        // Test min with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            auto result5 = torch::min(empty_tensor, 0, keepdim);
        } catch (const std::exception&) {
            // This might throw, which is expected behavior
        }
        
        // Test min with 0-dim tensor (scalar tensor)
        try {
            torch::Tensor scalar_tensor = torch::tensor(5);
            auto result6 = torch::min(scalar_tensor, 0, keepdim);
        } catch (const std::exception&) {
            // This might throw, which is expected behavior
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
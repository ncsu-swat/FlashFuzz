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
        
        // Extract dim parameter if we have more data
        int64_t dim = -1;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Apply torch.amin operation with different parameter combinations
        torch::Tensor result;
        
        // Case 1: No dimension specified (reduce over all dimensions)
        if (dim == -1 || input_tensor.dim() == 0) {
            result = torch::amin(input_tensor);
        }
        // Case 2: Specific dimension with keepdim option
        else {
            // Allow dim to be any value, including negative or out of bounds
            // Let PyTorch handle the validation
            result = torch::amin(input_tensor, dim, keepdim);
        }
        
        // Try another variant with multiple dimensions if tensor has enough dimensions
        if (input_tensor.dim() >= 2 && offset + sizeof(int64_t) <= Size) {
            std::vector<int64_t> dims;
            
            // Extract number of dimensions to reduce over
            int64_t num_dims = 1 + (Data[offset] % std::min(input_tensor.dim(), static_cast<int64_t>(3)));
            offset++;
            
            // Extract the dimensions
            for (int64_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; i++) {
                int64_t d;
                std::memcpy(&d, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dims.push_back(d);
            }
            
            // Apply amin with multiple dimensions
            if (!dims.empty()) {
                result = torch::amin(input_tensor, dims, keepdim);
            }
        }
        
        // Ensure the result is used to prevent optimization
        if (result.defined()) {
            volatile float dummy = 0.0;
            if (result.numel() > 0 && result.scalar_type() != torch::kBool) {
                dummy = result.item<float>();
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
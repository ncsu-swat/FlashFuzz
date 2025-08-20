#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for torch::unique
        bool sorted = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        bool return_inverse = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        bool return_counts = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        // Get dimension parameter if there's enough data
        int64_t dim = -1;
        bool has_dim = false;
        if (offset < Size) {
            has_dim = (Data[offset++] % 2 == 0);
            if (has_dim && offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // If tensor is not empty, ensure dim is within valid range
                if (input_tensor.dim() > 0) {
                    dim = dim % input_tensor.dim();
                    if (dim < 0) dim += input_tensor.dim();
                }
            }
        }
        
        // Call torch::unique with different parameter combinations
        if (has_dim) {
            if (return_inverse && return_counts) {
                auto [output, inverse_indices, counts] = torch::unique_dim(input_tensor, dim, sorted, return_inverse, return_counts);
            } else if (return_inverse) {
                auto [output, inverse_indices, counts] = torch::unique_dim(input_tensor, dim, sorted, return_inverse);
            } else if (return_counts) {
                auto [output, inverse_indices, counts] = torch::unique_dim(input_tensor, dim, sorted, false, return_counts);
            } else {
                auto output = torch::unique_dim(input_tensor, dim, sorted);
            }
        } else {
            if (return_inverse && return_counts) {
                auto [output, inverse_indices, counts] = torch::_unique2(input_tensor, sorted, return_inverse, return_counts);
            } else if (return_inverse) {
                auto [output, inverse_indices] = torch::_unique(input_tensor, sorted, return_inverse);
            } else if (return_counts) {
                auto [output, inverse_indices, counts] = torch::_unique2(input_tensor, sorted, false, return_counts);
            } else {
                auto [output, inverse_indices] = torch::_unique(input_tensor, sorted);
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
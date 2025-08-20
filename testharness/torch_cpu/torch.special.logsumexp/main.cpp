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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dim parameter if there's data left
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // For tensors with rank > 0, ensure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Parse keepdim parameter if there's data left
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Try different variants of logsumexp
        
        // Variant 1: Basic logsumexp
        torch::Tensor result1 = torch::special::logsumexp(input, dim, keepdim);
        
        // Variant 2: If we have a multi-dimensional tensor, try with a list of dimensions
        if (input.dim() > 1 && offset + sizeof(int64_t) <= Size) {
            int64_t dim2;
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            dim2 = dim2 % input.dim();
            if (dim2 < 0) {
                dim2 += input.dim();
            }
            
            // Ensure dim2 is different from dim
            if (dim2 == dim && input.dim() > 1) {
                dim2 = (dim2 + 1) % input.dim();
            }
            
            std::vector<int64_t> dims = {dim, dim2};
            torch::Tensor result2 = torch::special::logsumexp(input, dims, keepdim);
        }
        
        // Variant 3: Try with empty dimensions list for scalar output
        if (input.dim() > 0) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input.dim(); i++) {
                all_dims.push_back(i);
            }
            torch::Tensor result3 = torch::special::logsumexp(input, all_dims, keepdim);
        }
        
        // Variant 4: Try with no dimensions specified (should reduce over all dimensions)
        torch::Tensor result4 = torch::special::logsumexp(input, {}, keepdim);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
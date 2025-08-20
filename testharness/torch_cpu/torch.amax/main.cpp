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
        
        // Extract dim parameter if we have more data
        int64_t dim = -1;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative dimensions for testing edge cases
            if (input.dim() > 0) {
                dim = dim % (2 * input.dim()) - input.dim();
            }
        }
        
        // Extract keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Test different variants of amax
        
        // Variant 1: amax over all dimensions
        torch::Tensor result1 = torch::amax(input);
        
        // Variant 2: amax over specific dimension
        if (input.dim() > 0) {
            torch::Tensor result2 = torch::amax(input, dim, keepdim);
        }
        
        // Variant 3: amax over multiple dimensions (if tensor has at least 2 dimensions)
        if (input.dim() >= 2) {
            std::vector<int64_t> dims;
            
            // Create a list of dimensions to reduce over
            int num_dims = 1 + (Size % input.dim());
            for (int i = 0; i < num_dims; i++) {
                if (offset < Size) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                    dims.push_back(d);
                }
            }
            
            // Remove duplicates if any
            std::sort(dims.begin(), dims.end());
            dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
            
            if (!dims.empty()) {
                torch::Tensor result3 = torch::amax(input, dims, keepdim);
            }
        }
        
        // Variant 4: out variant
        if (input.dim() > 0) {
            // Create output tensor with appropriate shape
            std::vector<int64_t> out_shape;
            if (keepdim) {
                out_shape = input.sizes().vec();
                if (dim >= 0 && dim < input.dim()) {
                    out_shape[dim] = 1;
                }
            } else {
                for (int64_t i = 0; i < input.dim(); i++) {
                    if (i != dim) {
                        out_shape.push_back(input.size(i));
                    }
                }
            }
            
            if (!out_shape.empty()) {
                torch::Tensor output = torch::empty(out_shape, input.options());
                torch::amax_out(output, input, dim, keepdim);
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
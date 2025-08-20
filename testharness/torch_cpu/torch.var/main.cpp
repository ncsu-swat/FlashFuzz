#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

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
        
        // Extract parameters for var operation if we have more data
        bool unbiased = false;
        bool keepdim = false;
        
        if (offset + 1 < Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        if (offset + 1 < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Try different variants of torch::var
        
        // Variant 1: var over all dimensions
        torch::Tensor result1 = torch::var(input_tensor, unbiased);
        
        // Variant 2: var with keepdim
        torch::Tensor result2 = torch::var(input_tensor, unbiased, keepdim);
        
        // Variant 3: var along specific dimension if tensor has dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            // Get a dimension to reduce along
            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
            
            // Try negative dimension index
            if (offset < Size && (Data[offset++] & 0x1)) {
                dim = -dim - 1;
            }
            
            torch::Tensor result3 = torch::var(input_tensor, dim, unbiased, keepdim);
            
            // Try with a list of dimensions if tensor has multiple dimensions
            if (input_tensor.dim() > 1 && offset < Size) {
                int num_dims = Data[offset++] % input_tensor.dim() + 1;
                std::vector<int64_t> dims;
                
                for (int i = 0; i < num_dims && offset < Size; i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                    // Ensure no duplicate dimensions
                    if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                        dims.push_back(d);
                    }
                }
                
                if (!dims.empty()) {
                    torch::Tensor result4 = torch::var(input_tensor, dims, unbiased, keepdim);
                }
            }
        }
        
        // Variant 4: Named dimension if available
        if (input_tensor.dim() > 0 && input_tensor.names().size() > 0 && offset < Size) {
            torch::Dimname dim = input_tensor.names()[0];
            torch::Tensor result5 = torch::var(input_tensor, dim, unbiased, keepdim);
        }
        
        // Variant 5: Try with correction parameter (instead of unbiased)
        if (offset < Size) {
            int correction = static_cast<int>(Data[offset++]) % 10;
            torch::Tensor result6 = torch::var(input_tensor, at::IntArrayRef{}, correction, keepdim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
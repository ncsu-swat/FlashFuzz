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
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Apply nansum operation in different ways to maximize coverage
        torch::Tensor result;
        
        // Case 1: nansum over all dimensions
        result = torch::nansum(input_tensor);
        
        // Case 2: nansum with specified dimension
        if (input_tensor.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            int64_t actual_dim = (dim % input_tensor.dim());
            if (actual_dim < 0) actual_dim += input_tensor.dim();
            
            result = torch::nansum(input_tensor, actual_dim, keepdim);
            
            // Case 3: nansum with dimension and explicit keepdim=false
            result = torch::nansum(input_tensor, actual_dim, false);
            
            // Case 4: nansum with dimension and explicit keepdim=true
            result = torch::nansum(input_tensor, actual_dim, true);
        }
        
        // Case 5: nansum with dimension array if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            std::vector<int64_t> dims;
            
            // Create a list of dimensions to sum over
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                if ((i % 2) == 0) {  // Take some dimensions
                    dims.push_back(i);
                }
            }
            
            if (!dims.empty()) {
                result = torch::nansum(input_tensor, dims, keepdim);
            }
        }
        
        // Case 6: nansum with dtype specified
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with dtype (using nullopt for dim parameter)
            result = torch::nansum(input_tensor, std::nullopt, false, dtype);
            
            // Try with dim and dtype
            if (input_tensor.dim() > 0) {
                int64_t actual_dim = (dim % input_tensor.dim());
                if (actual_dim < 0) actual_dim += input_tensor.dim();
                
                result = torch::nansum(input_tensor, actual_dim, keepdim, dtype);
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
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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to squeeze if there's data left
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test squeeze_copy with no dimension specified
        torch::Tensor result1 = torch::squeeze_copy(input_tensor);
        
        // Test squeeze_copy with dimension specified
        torch::Tensor result2;
        if (input_tensor.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            result2 = torch::squeeze_copy(input_tensor, dim);
        }
        
        // Test squeeze_copy with dimension list
        if (offset + sizeof(int64_t) <= Size && input_tensor.dim() > 0) {
            int64_t num_dims_to_squeeze;
            std::memcpy(&num_dims_to_squeeze, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Limit to a reasonable number
            num_dims_to_squeeze = std::abs(num_dims_to_squeeze) % (input_tensor.dim() + 1);
            
            std::vector<int64_t> dims_to_squeeze;
            for (int64_t i = 0; i < num_dims_to_squeeze && offset + sizeof(int64_t) <= Size; i++) {
                int64_t squeeze_dim;
                std::memcpy(&squeeze_dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure dim is within valid range
                squeeze_dim = squeeze_dim % (2 * input_tensor.dim()) - input_tensor.dim();
                dims_to_squeeze.push_back(squeeze_dim);
            }
            
            // Test with dimension list if we have any dimensions
            if (!dims_to_squeeze.empty()) {
                torch::Tensor result3 = torch::squeeze_copy(input_tensor, dims_to_squeeze);
            }
        }
        
        // Test in-place version (squeeze_) on a copy of the input tensor
        torch::Tensor copy_tensor = input_tensor.clone();
        copy_tensor.squeeze_();
        
        // Test in-place version with dimension
        if (input_tensor.dim() > 0) {
            torch::Tensor copy_tensor2 = input_tensor.clone();
            dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            copy_tensor2.squeeze_(dim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter for argmin if we have more data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            // Extract dimension from the input data
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor is not empty, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                // Allow negative dimensions (PyTorch handles them by wrapping)
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            }
        }
        
        // Extract keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Apply argmin operation with different parameter combinations
        torch::Tensor result;
        
        // Test different variants of argmin
        if (offset % 3 == 0) {
            // Variant 1: argmin without parameters (reduces over all dimensions)
            result = torch::argmin(input_tensor);
        } 
        else if (offset % 3 == 1) {
            // Variant 2: argmin with dimension parameter
            result = torch::argmin(input_tensor, dim);
        }
        else {
            // Variant 3: argmin with dimension and keepdim parameters
            result = torch::argmin(input_tensor, dim, keepdim);
        }
        
        // Access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto value = result.item<int64_t>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

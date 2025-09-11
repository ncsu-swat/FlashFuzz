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
        
        // Get a value to determine if we should specify dims
        bool specify_dims = false;
        int64_t dim = 0;
        
        if (offset < Size) {
            specify_dims = Data[offset++] % 2 == 0;
            
            // If we have more data and we're specifying dims, get the dimension
            if (specify_dims && offset < Size) {
                // Get a dimension value, potentially negative or out of bounds
                uint8_t dim_byte = Data[offset++];
                dim = static_cast<int8_t>(dim_byte); // Convert to signed to allow negative dims
            }
        }
        
        // Apply ifftshift operation
        torch::Tensor result;
        if (specify_dims) {
            // Call ifftshift with specific dimension
            result = torch::fft::ifftshift(input_tensor, dim);
        } else {
            // Call ifftshift without dimension argument
            result = torch::fft::ifftshift(input_tensor);
        }
        
        // Basic validation - just check that the result has the same shape as input
        if (result.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("ifftshift result has different shape than input");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

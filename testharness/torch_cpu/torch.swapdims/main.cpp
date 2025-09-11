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
        
        // Need at least a few bytes for tensor creation and dimension indices
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor rank
        int64_t rank = input_tensor.dim();
        
        // If we've consumed all data or tensor is scalar, return
        if (offset + 2 > Size || rank < 2) {
            return 0;
        }
        
        // Extract dimension indices for swapping
        int64_t dim1 = static_cast<int8_t>(Data[offset++]); // Use signed int8_t to allow negative indices
        int64_t dim2 = static_cast<int8_t>(Data[offset++]);
        
        // Apply swapdims operation
        torch::Tensor result = torch::swapdims(input_tensor, dim1, dim2);
        
        // Verify result is not empty
        if (result.numel() != input_tensor.numel()) {
            throw std::runtime_error("Result tensor has different number of elements");
        }
        
        // Try another variant of the API
        result = input_tensor.swapdims(dim1, dim2);
        
        // Try with different dimensions if we have more data
        if (offset + 2 <= Size && rank >= 2) {
            dim1 = static_cast<int8_t>(Data[offset++]);
            dim2 = static_cast<int8_t>(Data[offset++]);
            result = torch::swapdims(input_tensor, dim1, dim2);
        }
        
        // Try the alias transpose (which is swapdims with 2 dimensions)
        if (rank >= 2) {
            result = torch::transpose(input_tensor, dim1, dim2);
            result = input_tensor.transpose(dim1, dim2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

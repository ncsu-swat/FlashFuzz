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
        
        // Need at least 1 byte for the dimension parameter
        if (Size < 1) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension parameter for unsqueeze
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a byte for the dimension
            uint8_t dim_byte = Data[offset++];
            
            // Convert to a dimension value that could be positive or negative
            int rank = input_tensor.dim();
            
            // Allow dim to be in range [-rank-1, rank]
            // This includes valid dimensions plus one position beyond each end
            dim = static_cast<int64_t>(dim_byte) % (2 * rank + 2) - (rank + 1);
        }
        
        // Apply unsqueeze operation
        torch::Tensor result = torch::unsqueeze(input_tensor, dim);
        
        // Verify the result has one more dimension than the input
        if (result.dim() != input_tensor.dim() + 1) {
            throw std::runtime_error("Unexpected result dimension");
        }
        
        // Try to access elements to ensure tensor is valid
        if (result.numel() > 0) {
            result.item();
        }
        
        // Try alternative API
        torch::Tensor result2 = input_tensor.unsqueeze(dim);
        
        // Try chained unsqueeze operations if we have more data
        if (offset < Size) {
            int64_t dim2 = static_cast<int64_t>(Data[offset++]) % (result.dim() + 1) - (result.dim() / 2);
            torch::Tensor result3 = result.unsqueeze(dim2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor
        torch::Tensor indices_tensor;
        if (offset < Size) {
            indices_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure indices are integers
            if (indices_tensor.scalar_type() != torch::kInt64 && 
                indices_tensor.scalar_type() != torch::kInt32 && 
                indices_tensor.scalar_type() != torch::kInt16 && 
                indices_tensor.scalar_type() != torch::kInt8) {
                indices_tensor = indices_tensor.to(torch::kInt64);
            }
        } else {
            // If we don't have enough data for a second tensor, create a simple indices tensor
            indices_tensor = torch::tensor({0, 1, -1}, torch::kInt64);
        }
        
        // Apply torch.take operation
        torch::Tensor result;
        
        // Try different variants of the take operation
        if (offset < Size && Data[offset] % 3 == 0) {
            // Basic take operation
            result = torch::take(input_tensor, indices_tensor);
        } else if (offset < Size && Data[offset] % 3 == 1) {
            // Take along dim if input is not a scalar
            if (input_tensor.dim() > 0) {
                int64_t dim = 0;
                if (offset + 1 < Size) {
                    dim = static_cast<int64_t>(Data[offset + 1]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
                }
                result = torch::take_along_dim(input_tensor, indices_tensor, dim);
            } else {
                result = torch::take(input_tensor, indices_tensor);
            }
        } else {
            // Take with potentially out-of-bounds indices to test edge cases
            result = torch::take(input_tensor, indices_tensor);
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely sum value encountered";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

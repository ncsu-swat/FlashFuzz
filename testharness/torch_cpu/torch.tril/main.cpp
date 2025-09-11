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
        
        // Parse diagonal parameter if we have more data
        int64_t diagonal = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&diagonal, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply torch.tril operation
        torch::Tensor result = torch::tril(input_tensor, diagonal);
        
        // Try another variant with different diagonal value if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t diagonal2;
            std::memcpy(&diagonal2, Data + offset, sizeof(int64_t));
            torch::Tensor result2 = torch::tril(input_tensor, diagonal2);
        }
        
        // Try in-place variant if possible
        if (input_tensor.is_floating_point() && input_tensor.is_contiguous()) {
            try {
                torch::Tensor input_copy = input_tensor.clone();
                input_copy.tril_(diagonal);
            } catch (const std::exception&) {
                // Ignore exceptions from in-place operation
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0, 0}, input_tensor.options());
            torch::Tensor empty_result = torch::tril(empty_tensor, diagonal);
        } catch (const std::exception&) {
            // Ignore exceptions from empty tensor
        }
        
        // Try with 1D tensor
        try {
            if (input_tensor.dim() > 0) {
                torch::Tensor tensor_1d = input_tensor.flatten();
                torch::Tensor result_1d = torch::tril(tensor_1d, diagonal);
            }
        } catch (const std::exception&) {
            // Ignore exceptions from 1D tensor
        }
        
        // Try with scalar tensor
        try {
            torch::Tensor scalar_tensor = torch::tensor(1.0, input_tensor.options());
            torch::Tensor scalar_result = torch::tril(scalar_tensor, diagonal);
        } catch (const std::exception&) {
            // Ignore exceptions from scalar tensor
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

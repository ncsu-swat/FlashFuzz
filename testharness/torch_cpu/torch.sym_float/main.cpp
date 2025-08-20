#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.sym_float operation - convert to float type
        torch::Tensor result = input_tensor.to(torch::kFloat);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the tensor
            if (result.numel() > 0) {
                result.item();
            }
        }
        
        // Try with different input types if we have more data
        if (offset + 2 < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor second_result = second_tensor.to(torch::kFloat);
            
            if (second_result.defined() && second_result.numel() > 0) {
                second_result.item();
            }
        }
        
        // Try with scalar inputs if we have more data
        if (offset < Size) {
            // Use remaining data to create a scalar value
            int64_t scalar_value = 0;
            size_t bytes_to_copy = std::min(sizeof(scalar_value), Size - offset);
            std::memcpy(&scalar_value, Data + offset, bytes_to_copy);
            
            // Apply float conversion to scalar
            torch::Scalar scalar(scalar_value);
            torch::Tensor scalar_result = torch::tensor(scalar).to(torch::kFloat);
            
            if (scalar_result.defined() && scalar_result.numel() > 0) {
                scalar_result.item();
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
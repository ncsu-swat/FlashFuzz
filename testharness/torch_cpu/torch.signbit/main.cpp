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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.signbit operation
        torch::Tensor result = torch::signbit(input_tensor);
        
        // Try some variations of the operation
        if (offset + 1 < Size) {
            // Try with out tensor
            torch::Tensor out_tensor = torch::empty_like(result);
            torch::signbit_out(out_tensor, input_tensor);
            
            // Try with non-empty tensor that has different shape
            if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                torch::Tensor different_shape = torch::empty({1}, input_tensor.options());
                try {
                    torch::signbit_out(different_shape, input_tensor);
                } catch (const std::exception&) {
                    // Expected to fail in some cases due to shape mismatch
                }
            }
        }
        
        // Try with different dtypes if we have enough data
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try to convert input tensor to a different dtype
            try {
                torch::Tensor converted = input_tensor.to(dtype);
                torch::Tensor result_converted = torch::signbit(converted);
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid
            }
        }
        
        // Try with a scalar input if we have enough data
        if (offset + sizeof(double) <= Size) {
            double scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            try {
                // Create a scalar tensor
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_result = torch::signbit(scalar_tensor);
            } catch (const std::exception&) {
                // Some scalar operations might fail
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::signbit(empty_tensor);
        } catch (const std::exception&) {
            // Empty tensor might cause issues
        }
        
        // Try with NaN and Inf values
        try {
            torch::Tensor special_values = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 
                                                         std::numeric_limits<float>::infinity(),
                                                         -std::numeric_limits<float>::infinity(),
                                                         0.0f, -0.0f});
            torch::Tensor special_result = torch::signbit(special_values);
        } catch (const std::exception&) {
            // Special values might cause issues
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

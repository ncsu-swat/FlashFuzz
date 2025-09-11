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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply cosh operation
        torch::Tensor result = torch::cosh(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.cosh_();
        }
        
        // Try with options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Use option byte to determine if we should test with specific dtype
            if (option_byte % 3 == 0) {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor result_with_dtype = torch::cosh(input.to(dtype));
            }
            
            // Test with named parameters
            if (option_byte % 5 == 0 && offset < Size) {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor result_named = torch::cosh(input.to(torch::dtype(dtype)));
            }
        }
        
        // Test edge cases with special values if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Create a tensor with special values
            std::vector<int64_t> shape = {1};
            torch::Tensor special_tensor;
            
            switch (edge_case % 5) {
                case 0: // Infinity
                    special_tensor = torch::full(shape, std::numeric_limits<float>::infinity());
                    break;
                case 1: // NaN
                    special_tensor = torch::full(shape, std::numeric_limits<float>::quiet_NaN());
                    break;
                case 2: // Very large value
                    special_tensor = torch::full(shape, 1e38);
                    break;
                case 3: // Very small value
                    special_tensor = torch::full(shape, 1e-38);
                    break;
                case 4: // Zero
                    special_tensor = torch::zeros(shape);
                    break;
            }
            
            // Apply cosh to the special tensor
            torch::Tensor special_result = torch::cosh(special_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

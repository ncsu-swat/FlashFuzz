#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the bessel_j1 operation
        torch::Tensor result = torch::special::bessel_j1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            result.item();
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset >= 2) {
            // Create another tensor with different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply bessel_j1 to the second tensor
            torch::Tensor result2 = torch::special::bessel_j1(input2);
            
            // Access the result
            if (result2.defined() && result2.numel() > 0) {
                result2.item();
            }
        }
        
        // Test with scalar input if we have more data
        if (Size - offset >= 1) {
            double scalar_value = static_cast<double>(Data[offset++]);
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
            torch::Tensor scalar_result = torch::special::bessel_j1(scalar_tensor);
            
            if (scalar_result.defined() && scalar_result.numel() > 0) {
                scalar_result.item();
            }
        }
        
        // Test with extreme values
        if (Size - offset >= 1) {
            uint8_t selector = Data[offset++] % 4;
            torch::Tensor extreme_tensor;
            
            switch (selector) {
                case 0:
                    extreme_tensor = torch::tensor(std::numeric_limits<float>::infinity());
                    break;
                case 1:
                    extreme_tensor = torch::tensor(-std::numeric_limits<float>::infinity());
                    break;
                case 2:
                    extreme_tensor = torch::tensor(std::numeric_limits<float>::quiet_NaN());
                    break;
                case 3:
                    extreme_tensor = torch::tensor(0.0);
                    break;
            }
            
            torch::Tensor extreme_result = torch::special::bessel_j1(extreme_tensor);
            
            if (extreme_result.defined() && extreme_result.numel() > 0) {
                extreme_result.item();
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
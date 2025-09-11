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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the spherical_bessel_j0 operation
        torch::Tensor result = torch::special::spherical_bessel_j0(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element;
        }
        
        // Try with different input types
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = torch::special::spherical_bessel_j0(input2);
        }
        
        // Try with scalar input
        if (offset + 1 < Size) {
            double scalar_input = static_cast<double>(Data[offset]) / 255.0;
            torch::Tensor scalar_tensor = torch::tensor(scalar_input);
            torch::Tensor scalar_result = torch::special::spherical_bessel_j0(scalar_tensor);
        }
        
        // Try with extreme values
        if (input.numel() > 0) {
            // Large values
            torch::Tensor large_input = input * 1e10;
            torch::Tensor large_result = torch::special::spherical_bessel_j0(large_input);
            
            // Small values
            torch::Tensor small_input = input * 1e-10;
            torch::Tensor small_result = torch::special::spherical_bessel_j0(small_input);
            
            // Negative values
            torch::Tensor neg_input = -input;
            torch::Tensor neg_result = torch::special::spherical_bessel_j0(neg_input);
            
            // NaN and Inf values
            torch::Tensor special_values = torch::tensor({0.0, INFINITY, -INFINITY, NAN});
            torch::Tensor special_result = torch::special::spherical_bessel_j0(special_values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

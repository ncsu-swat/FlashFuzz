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
        
        // Apply the scaled_modified_bessel_k1 operation
        torch::Tensor result = torch::special::scaled_modified_bessel_k1(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element;
        }
        
        // Try with different input types
        if (offset + 2 < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            torch::Tensor result2 = torch::special::scaled_modified_bessel_k1(input2);
        }
        
        // Try with scalar input converted to tensor
        if (input.numel() > 0) {
            torch::Scalar scalar_input = input.item();
            try {
                torch::Tensor scalar_tensor = torch::tensor(scalar_input);
                torch::Tensor scalar_result = torch::special::scaled_modified_bessel_k1(scalar_tensor);
            } catch (...) {
                // Ignore exceptions from scalar input
            }
        }
        
        // Try with different device if available
        try {
            if (torch::cuda::is_available()) {
                torch::Tensor cuda_input = input.cuda();
                torch::Tensor cuda_result = torch::special::scaled_modified_bessel_k1(cuda_input);
            }
        } catch (...) {
            // Ignore CUDA-related exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

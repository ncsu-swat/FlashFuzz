#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract eps value from the remaining data if available
        double eps = 1e-6; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is in a reasonable range
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-6;
            if (std::isnan(eps) || std::isinf(eps)) eps = 1e-6;
            // Clamp to reasonable range
            if (eps > 0.5) eps = 0.5;
        }
        
        // Make a copy of the input tensor to preserve original data
        // logit_ requires floating point tensor
        torch::Tensor tensor_copy = input_tensor.to(torch::kFloat32).clone();
        
        // Apply logit_ in-place operation
        // logit(x) = log(x / (1 - x)), expects input in range [0, 1]
        // eps clamps input to [eps, 1-eps] to avoid inf
        tensor_copy.logit_(eps);
        
        // Also test the functional version for better coverage
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_float = input_tensor.to(torch::kFloat32);
            torch::Tensor result = torch::logit(input_float, eps);
            (void)result; // Prevent unused variable warning
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - arcsinh_ works on floating point tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // arcsinh_ requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Make a copy for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arcsinh_ operation (in-place)
        // arcsinh_(x) computes the inverse hyperbolic sine in-place
        input_copy.arcsinh_();
        
        // Also test the non-in-place version for coverage
        torch::Tensor result = torch::arcsinh(input);
        
        // Access results to ensure computation happens
        (void)input_copy.sum().item<float>();
        (void)result.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
}
#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bessel_j0 requires floating-point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply the bessel_j0 operation
        torch::Tensor result = torch::special::bessel_j0(input);
        
        // Access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try with different input configurations if we have more data
        if (Size - offset >= 4) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat64);  // Test with double precision
            }
            
            torch::Tensor result2 = torch::special::bessel_j0(input2);
            
            if (result2.defined() && result2.numel() > 0) {
                volatile double sum = result2.sum().item<double>();
                (void)sum;
            }
        }
        
        // Test with edge cases - extreme values
        if (Size - offset >= 2) {
            torch::Tensor edge_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!edge_input.is_floating_point()) {
                edge_input = edge_input.to(torch::kFloat32);
            }
            
            // Scale to create extreme values
            edge_input = edge_input * 1e6f;
            
            torch::Tensor edge_result = torch::special::bessel_j0(edge_input);
            
            if (edge_result.defined() && edge_result.numel() > 0) {
                volatile float sum = edge_result.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test with special values: zeros
        {
            torch::Tensor zeros = torch::zeros({2, 2}, torch::kFloat32);
            torch::Tensor zero_result = torch::special::bessel_j0(zeros);
            volatile float sum = zero_result.sum().item<float>();
            (void)sum;
        }
        
        // Test with negative values
        if (Size >= 1) {
            int neg_size = (Data[0] % 4) + 1;
            torch::Tensor neg_input = torch::randn({neg_size, neg_size}, torch::kFloat32) * -10.0f;
            torch::Tensor neg_result = torch::special::bessel_j0(neg_input);
            volatile float sum = neg_result.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
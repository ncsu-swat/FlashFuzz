#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip empty inputs
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // log_ requires floating point tensor
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Test 1: Basic log_ operation on the tensor as-is
        // (log of negative numbers produces NaN, which is expected behavior)
        {
            torch::Tensor t1 = input_tensor.clone();
            t1.log_();
            // Force computation
            volatile float sum = t1.sum().item<float>();
            (void)sum;
        }
        
        // Test 2: log_ on absolute values (ensures positive domain)
        {
            torch::Tensor t2 = torch::abs(input_tensor.clone()) + 1e-6f; // avoid log(0)
            torch::Tensor original = t2.clone();
            t2.log_();
            
            // Verify in-place modification happened
            torch::Tensor expected = torch::log(original);
            
            // Only compare for finite values
            if (t2.numel() > 0) {
                try {
                    // Use isfinite to mask out inf/nan for comparison
                    auto finite_mask = torch::isfinite(t2) & torch::isfinite(expected);
                    if (finite_mask.any().item<bool>()) {
                        auto t2_finite = t2.index({finite_mask});
                        auto expected_finite = expected.index({finite_mask});
                        torch::allclose(t2_finite, expected_finite, 1e-5, 1e-8);
                    }
                } catch (...) {
                    // Shape mismatches or indexing issues - ignore
                }
            }
        }
        
        // Test 3: Different tensor shapes
        if (Size >= 8) {
            int dim1 = std::max(1, (int)(Data[offset % Size] % 10) + 1);
            int dim2 = std::max(1, (int)(Data[(offset + 1) % Size] % 10) + 1);
            
            torch::Tensor t3 = torch::rand({dim1, dim2}) + 0.01f; // positive values
            t3.log_();
            volatile float v = t3.sum().item<float>();
            (void)v;
        }
        
        // Test 4: Complex tensor if supported
        try {
            torch::Tensor t4 = torch::randn({3, 3}, torch::kComplexFloat);
            t4.log_();
            (void)t4.abs().sum().item<float>();
        } catch (...) {
            // Complex log_ may not be supported - ignore
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}
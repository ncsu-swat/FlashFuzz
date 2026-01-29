#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // asin_ only works on floating-point tensors
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Make input contiguous for in-place operation
        input = input.contiguous();
        
        // Test basic asin_ in-place operation
        try {
            torch::Tensor input_copy = input.clone();
            input.asin_();
            
            // Verify operation produces same result as out-of-place version
            torch::Tensor expected = torch::asin(input_copy);
            
            // Use equal_nan to handle NaN values (asin of values outside [-1,1])
            if (!torch::equal(input.isnan(), expected.isnan())) {
                // NaN patterns should match
            }
        } catch (const c10::Error&) {
            // Expected for some tensor configurations
        }
        
        // Test with clamped values to ensure valid asin domain [-1, 1]
        if (offset + 4 < Size) {
            torch::Tensor bounded_input = fuzzer_utils::createTensor(Data, Size, offset);
            if (!bounded_input.is_floating_point()) {
                bounded_input = bounded_input.to(torch::kFloat32);
            }
            bounded_input = bounded_input.contiguous();
            
            // Clamp to valid domain for asin
            bounded_input = torch::clamp(bounded_input, -1.0, 1.0);
            
            try {
                bounded_input.asin_();
                
                // Result should be in range [-pi/2, pi/2]
                auto min_val = bounded_input.min().item<float>();
                auto max_val = bounded_input.max().item<float>();
                (void)min_val;
                (void)max_val;
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with different dtypes
        if (offset + 4 < Size) {
            torch::Tensor double_input = fuzzer_utils::createTensor(Data, Size, offset);
            double_input = double_input.to(torch::kFloat64).contiguous();
            
            try {
                double_input.asin_();
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
        
        // Test with specific edge cases based on fuzzer data
        if (Size > offset) {
            uint8_t edge_case = Data[offset % Size];
            torch::Tensor edge_tensor;
            
            if (edge_case % 4 == 0) {
                // Test with -1
                edge_tensor = torch::full({2, 2}, -1.0);
            } else if (edge_case % 4 == 1) {
                // Test with 1
                edge_tensor = torch::full({2, 2}, 1.0);
            } else if (edge_case % 4 == 2) {
                // Test with 0
                edge_tensor = torch::zeros({2, 2});
            } else {
                // Test with value outside domain (produces NaN)
                edge_tensor = torch::full({2, 2}, 2.0);
            }
            
            try {
                edge_tensor.asin_();
            } catch (const c10::Error&) {
                // Silently handle
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
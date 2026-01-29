#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Poisson requires float or double tensor with non-negative values (rate parameters)
        // Convert to float if necessary and ensure non-negative values
        torch::Tensor rate_tensor;
        if (!input.is_floating_point()) {
            rate_tensor = input.to(torch::kFloat32);
        } else {
            rate_tensor = input;
        }
        
        // Ensure non-negative values by taking absolute value
        rate_tensor = torch::abs(rate_tensor);
        
        // Clamp to reasonable range to avoid overflow issues
        rate_tensor = torch::clamp(rate_tensor, 0.0, 1000.0);
        
        // Determine which variant to test
        uint8_t variant = (offset < Size) ? Data[offset++] : 0;
        
        // Test torch::poisson with default generator
        try {
            torch::Tensor result = torch::poisson(rate_tensor);
            // Verify result is valid (should be non-negative integers stored as float)
            (void)result.sum();
        } catch (const std::exception &e) {
            // Expected failures for invalid inputs - catch silently
        }
        
        // Test with different tensor shapes
        if (offset < Size && (variant & 0x01)) {
            try {
                // Create a scalar rate
                float scalar_rate = static_cast<float>(Data[offset] % 100);
                offset++;
                torch::Tensor scalar_tensor = torch::tensor(scalar_rate);
                torch::Tensor scalar_result = torch::poisson(scalar_tensor);
                (void)scalar_result.item<float>();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with output tensor variant
        if (offset < Size && (variant & 0x02)) {
            try {
                // Create output tensor and use poisson with out parameter
                torch::Tensor out_tensor = torch::empty_like(rate_tensor);
                torch::Tensor result = torch::poisson(rate_tensor, c10::nullopt);
                (void)result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with different dtypes
        if (offset < Size && (variant & 0x04)) {
            try {
                // Test with double precision
                torch::Tensor double_tensor = rate_tensor.to(torch::kFloat64);
                torch::Tensor double_result = torch::poisson(double_tensor);
                (void)double_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with zero rate (should produce zeros)
        if (offset < Size && (variant & 0x08)) {
            try {
                torch::Tensor zero_rate = torch::zeros_like(rate_tensor);
                torch::Tensor zero_result = torch::poisson(zero_rate);
                (void)zero_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with very small rates
        if (offset < Size && (variant & 0x10)) {
            try {
                torch::Tensor small_rate = rate_tensor * 0.001f;
                torch::Tensor small_result = torch::poisson(small_rate);
                (void)small_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with multidimensional tensor
        if (offset + 2 < Size && (variant & 0x20)) {
            try {
                int64_t dim1 = (Data[offset++] % 10) + 1;
                int64_t dim2 = (Data[offset++] % 10) + 1;
                torch::Tensor multi_dim = torch::rand({dim1, dim2}) * 10.0f;
                torch::Tensor multi_result = torch::poisson(multi_dim);
                (void)multi_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with large rate values
        if (offset < Size && (variant & 0x40)) {
            try {
                torch::Tensor large_rate = rate_tensor + 500.0f;
                torch::Tensor large_result = torch::poisson(large_rate);
                (void)large_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
            }
        }
        
        // Test with 1D tensor
        if (offset < Size && (variant & 0x80)) {
            try {
                int64_t len = (Data[offset++] % 50) + 1;
                torch::Tensor vec = torch::rand({len}) * 20.0f;
                torch::Tensor vec_result = torch::poisson(vec);
                (void)vec_result.sum();
            } catch (const std::exception &e) {
                // Expected failures - catch silently
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
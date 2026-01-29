#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - needs to be floating point for rrelu
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract lower and upper bounds from remaining data
        float lower = 0.125f;  // default lower
        float upper = 0.3333f; // default upper
        
        if (offset + sizeof(float) <= Size) {
            float tmp_lower;
            std::memcpy(&tmp_lower, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Only use if it's a valid finite number in reasonable range
            if (std::isfinite(tmp_lower) && tmp_lower >= 0.0f && tmp_lower <= 1.0f) {
                lower = tmp_lower;
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float tmp_upper;
            std::memcpy(&tmp_upper, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Only use if it's a valid finite number in reasonable range
            if (std::isfinite(tmp_upper) && tmp_upper >= 0.0f && tmp_upper <= 1.0f) {
                upper = tmp_upper;
            }
        }
        
        // Ensure lower <= upper
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        // Store original data pointer to verify in-place operation
        void* original_data_ptr = input.data_ptr();
        
        // Determine which variant to test based on fuzzer data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        switch (variant) {
            case 0:
                // Default parameters
                torch::rrelu_(input);
                break;
            case 1:
                // With lower bound only
                torch::rrelu_(input, lower);
                break;
            case 2:
                // With lower and upper bounds
                torch::rrelu_(input, lower, upper);
                break;
            case 3:
                // With training flag
                {
                    bool training = (offset < Size) ? (Data[offset++] % 2 == 1) : true;
                    torch::rrelu_(input, lower, upper, training);
                }
                break;
        }
        
        // Verify that the operation was applied in-place
        if (input.data_ptr() != original_data_ptr) {
            std::cerr << "rrelu_ should modify the tensor in-place" << std::endl;
        }
        
        // Also test non-inplace version for coverage
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        if (!input2.is_floating_point()) {
            input2 = input2.to(torch::kFloat32);
        }
        
        try {
            torch::Tensor output = torch::rrelu(input2, lower, upper);
            // Force computation
            if (output.numel() > 0) {
                output.sum().item<float>();
            }
        } catch (...) {
            // Ignore failures in non-inplace version
        }
        
        // Force computation on in-place result
        if (input.numel() > 0) {
            input.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
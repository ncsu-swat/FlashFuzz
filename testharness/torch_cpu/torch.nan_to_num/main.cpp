#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

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
        
        // Need minimum data for tensor creation
        if (Size < 4) {
            return -1;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse replacement values for nan, posinf, neginf
        double nan_replacement = 0.0;
        double posinf_replacement = std::numeric_limits<double>::max();
        double neginf_replacement = std::numeric_limits<double>::lowest();
        
        // Parse nan_replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&nan_replacement, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize to avoid NaN as replacement
            if (std::isnan(nan_replacement) || std::isinf(nan_replacement)) {
                nan_replacement = 0.0;
            }
        }
        
        // Parse posinf_replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&posinf_replacement, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isnan(posinf_replacement) || std::isinf(posinf_replacement)) {
                posinf_replacement = std::numeric_limits<double>::max();
            }
        }
        
        // Parse neginf_replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&neginf_replacement, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isnan(neginf_replacement) || std::isinf(neginf_replacement)) {
                neginf_replacement = std::numeric_limits<double>::lowest();
            }
        }
        
        // Case 1: Default parameters (uses default replacement values)
        torch::Tensor result1 = torch::nan_to_num(input_tensor);
        
        // Case 2: With custom nan replacement
        torch::Tensor result2 = torch::nan_to_num(input_tensor, nan_replacement);
        
        // Case 3: With nan and posinf replacement
        torch::Tensor result3 = torch::nan_to_num(input_tensor, nan_replacement, posinf_replacement);
        
        // Case 4: With all replacement values
        torch::Tensor result4 = torch::nan_to_num(input_tensor, nan_replacement, posinf_replacement, neginf_replacement);
        
        // In-place version on a clone
        torch::Tensor input_copy = input_tensor.clone();
        torch::nan_to_num_(input_copy, nan_replacement, posinf_replacement, neginf_replacement);
        
        // Test with floating point tensors containing special values
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            try {
                // Create tensor with special values (NaN, +Inf, -Inf)
                torch::Tensor special_tensor = input_tensor.clone();
                auto flat = special_tensor.flatten();
                int64_t numel = flat.numel();
                
                if (numel >= 1) {
                    flat[0] = std::numeric_limits<double>::quiet_NaN();
                }
                if (numel >= 2) {
                    flat[1] = std::numeric_limits<double>::infinity();
                }
                if (numel >= 3) {
                    flat[2] = -std::numeric_limits<double>::infinity();
                }
                
                // Apply nan_to_num to the special tensor
                torch::Tensor special_result = torch::nan_to_num(special_tensor, nan_replacement, posinf_replacement, neginf_replacement);
                
                // In-place on special tensor
                torch::nan_to_num_(special_tensor);
            } catch (const std::exception &) {
                // Silently ignore expected failures
            }
        }
        
        // Test with different floating point dtypes
        if (input_tensor.numel() > 0) {
            std::vector<torch::ScalarType> float_types = {
                torch::kFloat, torch::kDouble
            };
            
            for (auto dtype : float_types) {
                try {
                    torch::Tensor converted = input_tensor.to(dtype);
                    torch::Tensor dtype_result = torch::nan_to_num(converted, nan_replacement, posinf_replacement, neginf_replacement);
                } catch (const std::exception &) {
                    // Silently ignore type conversion failures
                }
            }
            
            // Try half precision types separately (may not be supported on all platforms)
            try {
                torch::Tensor half_tensor = input_tensor.to(torch::kHalf);
                torch::nan_to_num(half_tensor);
            } catch (const std::exception &) {
                // Silently ignore
            }
            
            try {
                torch::Tensor bf16_tensor = input_tensor.to(torch::kBFloat16);
                torch::nan_to_num(bf16_tensor);
            } catch (const std::exception &) {
                // Silently ignore
            }
        }
        
        // Test with scalar tensor
        try {
            torch::Tensor scalar_nan = torch::tensor(std::numeric_limits<double>::quiet_NaN());
            torch::Tensor scalar_result = torch::nan_to_num(scalar_nan, nan_replacement);
            
            torch::Tensor scalar_inf = torch::tensor(std::numeric_limits<double>::infinity());
            torch::Tensor inf_result = torch::nan_to_num(scalar_inf, nan_replacement, posinf_replacement);
            
            torch::Tensor scalar_neginf = torch::tensor(-std::numeric_limits<double>::infinity());
            torch::Tensor neginf_result = torch::nan_to_num(scalar_neginf, nan_replacement, posinf_replacement, neginf_replacement);
        } catch (const std::exception &) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
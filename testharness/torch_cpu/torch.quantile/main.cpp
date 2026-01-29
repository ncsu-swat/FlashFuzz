#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        
        // Create input tensor - needs to be float type for quantile
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if needed (quantile requires floating point)
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Flatten empty tensors case
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Parse q value (quantile value between 0 and 1)
        float q = 0.5f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&q, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN and Inf, ensure q is between 0 and 1
            if (std::isnan(q) || std::isinf(q)) {
                q = 0.5f;
            } else {
                q = std::abs(q);
                q = q - std::floor(q);  // Keeps it in [0, 1)
            }
        }
        
        // Parse dim value - bound to valid range
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            int64_t ndim = input_tensor.dim();
            if (ndim > 0) {
                dim = raw_dim % ndim;
                if (dim < 0) dim += ndim;
            }
        }
        
        // Parse keepdim value
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Parse interpolation mode
        std::string interpolation = "linear";
        if (offset < Size) {
            uint8_t interp_selector = Data[offset++] % 5;
            switch (interp_selector) {
                case 0: interpolation = "linear"; break;
                case 1: interpolation = "lower"; break;
                case 2: interpolation = "higher"; break;
                case 3: interpolation = "midpoint"; break;
                case 4: interpolation = "nearest"; break;
            }
        }
        
        // Variant 1: Basic quantile with scalar q (reduces to single value)
        try {
            torch::Tensor result1 = torch::quantile(input_tensor, q);
            (void)result1;
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Variant 2: Quantile with specified dimension
        try {
            torch::Tensor result2 = torch::quantile(input_tensor, q, dim, keepdim);
            (void)result2;
        } catch (const c10::Error& e) {
            // Expected for shape mismatches
        }
        
        // Variant 3: Quantile with interpolation mode (no dim specified)
        try {
            torch::Tensor result3 = torch::quantile(input_tensor, q, c10::nullopt, false, interpolation);
            (void)result3;
        } catch (const c10::Error& e) {
            // Expected for invalid interpolation
        }
        
        // Variant 4: Full quantile with all parameters
        try {
            torch::Tensor result4 = torch::quantile(input_tensor, q, dim, keepdim, interpolation);
            (void)result4;
        } catch (const c10::Error& e) {
            // Expected for various invalid combinations
        }
        
        // Variant 5: Try with tensor q (multiple quantiles)
        try {
            std::vector<float> q_values = {0.25f, 0.5f, 0.75f};
            torch::Tensor q_tensor = torch::tensor(q_values);
            torch::Tensor result5 = torch::quantile(input_tensor, q_tensor);
            (void)result5;
        } catch (const c10::Error& e) {
            // Expected for various invalid inputs
        }
        
        // Variant 6: Tensor q with dimension
        try {
            std::vector<float> q_values = {0.1f, 0.9f};
            torch::Tensor q_tensor = torch::tensor(q_values);
            torch::Tensor result6 = torch::quantile(input_tensor, q_tensor, dim, keepdim, interpolation);
            (void)result6;
        } catch (const c10::Error& e) {
            // Expected
        }
        
        // Variant 7: Single element q tensor
        try {
            torch::Tensor q_tensor = torch::tensor({q});
            torch::Tensor result7 = torch::quantile(input_tensor, q_tensor, dim, keepdim);
            (void)result7;
        } catch (const c10::Error& e) {
            // Expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
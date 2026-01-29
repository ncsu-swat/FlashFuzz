#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor - if not enough data, clone tensor1 with slight modification
        torch::Tensor tensor2;
        if (offset >= Size) {
            tensor2 = tensor1.clone();
        } else {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Parse parameters for isclose from remaining data
        double rtol = 1e-5;
        double atol = 1e-8;
        bool equal_nan = false;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&rtol, Data + offset, sizeof(double));
            offset += sizeof(double);
            rtol = std::abs(rtol);
            // Clamp to reasonable range to avoid extremely large values
            if (std::isnan(rtol) || std::isinf(rtol) || rtol > 1e10) {
                rtol = 1e-5;
            }
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&atol, Data + offset, sizeof(double));
            offset += sizeof(double);
            atol = std::abs(atol);
            if (std::isnan(atol) || std::isinf(atol) || atol > 1e10) {
                atol = 1e-8;
            }
        }
        
        if (offset < Size) {
            equal_nan = Data[offset++] & 0x1;
        }
        
        // Call isclose with various parameter combinations
        try {
            torch::Tensor result1 = torch::isclose(tensor1, tensor2);
        } catch (const std::exception &) {
            // Shape mismatch or dtype issues - silently ignore
        }
        
        try {
            torch::Tensor result2 = torch::isclose(tensor1, tensor2, rtol, atol);
        } catch (const std::exception &) {
        }
        
        try {
            torch::Tensor result3 = torch::isclose(tensor1, tensor2, rtol, atol, equal_nan);
        } catch (const std::exception &) {
        }
        
        // Try with broadcasting - create scalar from one element
        if (tensor1.dim() > 0 && tensor1.numel() > 0) {
            try {
                torch::Tensor scalar_tensor = tensor1.flatten().index({0}).unsqueeze(0);
                torch::Tensor broadcast_result = torch::isclose(tensor1, scalar_tensor);
            } catch (const std::exception &) {
            }
        }
        
        // Try with different dtypes - convert both to float
        try {
            torch::Tensor float_tensor1 = tensor1.to(torch::kFloat);
            torch::Tensor float_tensor2 = tensor2.to(torch::kFloat);
            torch::Tensor result_float = torch::isclose(float_tensor1, float_tensor2);
        } catch (const std::exception &) {
        }
        
        // Try with double precision
        try {
            torch::Tensor double_tensor1 = tensor1.to(torch::kDouble);
            torch::Tensor double_tensor2 = tensor2.to(torch::kDouble);
            torch::Tensor result_double = torch::isclose(double_tensor1, double_tensor2, rtol, atol);
        } catch (const std::exception &) {
        }
        
        // Test with NaN values if we have floating point tensors
        if (tensor1.is_floating_point() && tensor1.numel() > 0) {
            try {
                torch::Tensor nan_tensor1 = tensor1.clone();
                torch::Tensor nan_tensor2 = tensor1.clone();
                
                nan_tensor1.flatten().index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                nan_tensor2.flatten().index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                
                // Test with equal_nan=false (NaN != NaN)
                torch::Tensor nan_result1 = torch::isclose(nan_tensor1, nan_tensor2, rtol, atol, false);
                // Test with equal_nan=true (NaN == NaN)
                torch::Tensor nan_result2 = torch::isclose(nan_tensor1, nan_tensor2, rtol, atol, true);
            } catch (const std::exception &) {
            }
        }
        
        // Test with infinity values
        if (tensor1.is_floating_point() && tensor1.numel() > 0) {
            try {
                torch::Tensor inf_tensor1 = tensor1.clone();
                torch::Tensor inf_tensor2 = tensor1.clone();
                
                inf_tensor1.flatten().index_put_({0}, std::numeric_limits<float>::infinity());
                inf_tensor2.flatten().index_put_({0}, std::numeric_limits<float>::infinity());
                
                torch::Tensor inf_result = torch::isclose(inf_tensor1, inf_tensor2);
            } catch (const std::exception &) {
            }
        }
        
        // Test comparing tensor with itself (should always be close)
        try {
            torch::Tensor self_result = torch::isclose(tensor1, tensor1);
        } catch (const std::exception &) {
        }
        
        // Test with zero tolerance
        try {
            torch::Tensor strict_result = torch::isclose(tensor1, tensor2, 0.0, 0.0);
        } catch (const std::exception &) {
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
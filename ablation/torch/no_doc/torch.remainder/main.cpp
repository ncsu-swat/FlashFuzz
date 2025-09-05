#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for creating tensors
        if (Size < 8) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Parse operation mode from first byte
        uint8_t op_mode = Data[offset++];
        
        // Create dividend tensor
        torch::Tensor dividend;
        try {
            dividend = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create first tensor, try with random small tensor
            dividend = torch::randn({2, 3});
        }
        
        // Create divisor - could be tensor or scalar
        bool use_scalar_divisor = (offset < Size) && (Data[offset++] % 3 == 0);
        
        torch::Tensor result;
        
        if (use_scalar_divisor && offset < Size) {
            // Use scalar divisor
            double scalar_divisor;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_divisor, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else {
                // Use remaining bytes to create a value
                scalar_divisor = (offset < Size) ? static_cast<double>(Data[offset++]) : 1.0;
            }
            
            // Handle special scalar values
            if (op_mode % 5 == 0) scalar_divisor = 0.0;  // Test division by zero
            else if (op_mode % 7 == 0) scalar_divisor = -scalar_divisor;  // Test negative
            else if (op_mode % 11 == 0) scalar_divisor = std::numeric_limits<double>::infinity();
            else if (op_mode % 13 == 0) scalar_divisor = std::numeric_limits<double>::quiet_NaN();
            else if (op_mode % 17 == 0) scalar_divisor = std::numeric_limits<double>::epsilon();
            
            // Test scalar remainder
            result = torch::remainder(dividend, scalar_divisor);
            
            // Also test reverse order if we have more data
            if (offset < Size && Data[offset++] % 2 == 0) {
                auto result2 = torch::remainder(scalar_divisor, dividend);
            }
        } else {
            // Use tensor divisor
            torch::Tensor divisor;
            try {
                divisor = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception& e) {
                // Create a compatible tensor
                divisor = torch::ones_like(dividend);
                if (offset < Size) {
                    // Add some variation based on remaining data
                    divisor = divisor * (1.0 + (Data[offset++] % 10) / 10.0);
                }
            }
            
            // Inject special values based on op_mode
            if (op_mode % 3 == 0 && divisor.numel() > 0) {
                // Inject zeros at random positions
                auto mask = torch::rand_like(divisor) < 0.2;
                divisor = torch::where(mask, torch::zeros_like(divisor), divisor);
            }
            if (op_mode % 5 == 1 && divisor.numel() > 0) {
                // Inject infinities
                auto mask = torch::rand_like(divisor) < 0.1;
                divisor = torch::where(mask, torch::full_like(divisor, std::numeric_limits<float>::infinity()), divisor);
            }
            if (op_mode % 7 == 1 && divisor.numel() > 0) {
                // Inject NaNs
                auto mask = torch::rand_like(divisor) < 0.1;
                divisor = torch::where(mask, torch::full_like(divisor, std::numeric_limits<float>::quiet_NaN()), divisor);
            }
            
            // Test broadcasting scenarios
            if (offset < Size && Data[offset++] % 4 == 0) {
                // Try to reshape for broadcasting
                if (dividend.dim() > 0 && divisor.dim() > 0) {
                    auto dividend_sizes = dividend.sizes().vec();
                    auto divisor_sizes = divisor.sizes().vec();
                    
                    // Make divisor broadcastable by setting some dims to 1
                    for (size_t i = 0; i < divisor_sizes.size() && offset < Size; ++i) {
                        if (Data[offset++] % 3 == 0) {
                            divisor_sizes[i] = 1;
                        }
                    }
                    try {
                        divisor = divisor.reshape(divisor_sizes);
                    } catch (...) {
                        // Keep original shape if reshape fails
                    }
                }
            }
            
            // Perform remainder operation
            result = torch::remainder(dividend, divisor);
            
            // Test in-place operation if possible
            if (offset < Size && Data[offset++] % 2 == 0) {
                try {
                    dividend.remainder_(divisor);
                } catch (...) {
                    // In-place might fail for various reasons (dtype, size mismatch, etc.)
                }
            }
        }
        
        // Additional operations to increase coverage
        if (result.defined() && offset < Size) {
            uint8_t extra_op = Data[offset++];
            
            // Test output tensor variant
            if (extra_op % 3 == 0) {
                torch::Tensor out = torch::empty_like(result);
                torch::remainder_out(out, dividend, use_scalar_divisor ? 
                    torch::scalar_tensor(1.5) : dividend);
            }
            
            // Test with different memory layouts
            if (extra_op % 5 == 0 && dividend.dim() >= 2) {
                auto transposed = dividend.transpose(0, 1);
                auto result_t = torch::remainder(transposed, 2.0);
            }
            
            // Test with non-contiguous tensors
            if (extra_op % 7 == 0 && dividend.numel() > 1) {
                auto strided = dividend.as_strided({1}, {2});
                auto result_s = torch::remainder(strided, 1.5);
            }
            
            // Test edge case: empty tensors
            if (extra_op % 11 == 0) {
                auto empty_tensor = torch::empty({0, 3});
                auto result_e = torch::remainder(empty_tensor, 1.0);
            }
            
            // Test with complex numbers if dtype supports
            if (extra_op % 13 == 0) {
                try {
                    auto complex_dividend = torch::complex(dividend, dividend);
                    auto result_c = torch::remainder(complex_dividend, 2.0);
                } catch (...) {
                    // Complex remainder might not be supported
                }
            }
        }
        
        // Validate result properties
        if (result.defined()) {
            // Access some properties to ensure tensor is valid
            auto shape = result.sizes();
            auto dtype = result.dtype();
            auto device = result.device();
            
            // Check for NaN/Inf in result
            if (result.dtype().isFloatingPoint()) {
                auto has_nan = torch::any(torch::isnan(result));
                auto has_inf = torch::any(torch::isinf(result));
            }
            
            // Force computation if lazy
            if (result.numel() > 0 && result.numel() < 1000) {
                result.cpu();  // Force materialization
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;  // Keep the input for coverage
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected errors
    }
    catch (...)
    {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}
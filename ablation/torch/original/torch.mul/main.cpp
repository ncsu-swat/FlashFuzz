#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Minimum size check: need at least 3 bytes (mode + 2 tensors/scalar metadata)
        if (Size < 3)
        {
            return 0;
        }

        // Parse multiplication mode
        uint8_t mul_mode = Data[offset++];
        bool use_scalar = (mul_mode % 3) == 0; // 1/3 chance to use scalar multiplication
        bool use_out_tensor = (mul_mode % 5) == 0; // 1/5 chance to use out parameter
        
        // Create first tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor result;
        torch::Tensor out_tensor;
        
        if (use_scalar)
        {
            // Scalar multiplication path
            if (offset >= Size)
            {
                // Use default scalar if no more data
                result = torch::mul(input, 2.5);
            }
            else
            {
                // Parse scalar value from remaining bytes
                double scalar_value = 1.0;
                if (offset + sizeof(double) <= Size)
                {
                    std::memcpy(&scalar_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    
                    // Bound scalar to reasonable range to avoid inf/nan
                    if (std::isnan(scalar_value) || std::isinf(scalar_value))
                    {
                        scalar_value = 1.0;
                    }
                    else
                    {
                        // Clamp to [-1e6, 1e6] range
                        scalar_value = std::max(-1e6, std::min(1e6, scalar_value));
                    }
                }
                else if (offset < Size)
                {
                    // Use single byte as scalar
                    scalar_value = static_cast<double>(Data[offset++]) / 127.5 - 1.0; // Range [-1, 1]
                }
                
                if (use_out_tensor && offset < Size)
                {
                    try
                    {
                        // Create output tensor with same shape as input
                        out_tensor = torch::empty_like(input);
                        torch::mul_out(out_tensor, input, scalar_value);
                        result = out_tensor;
                    }
                    catch (const c10::Error &e)
                    {
                        // Fall back to regular multiplication if out fails
                        result = torch::mul(input, scalar_value);
                    }
                }
                else
                {
                    result = torch::mul(input, scalar_value);
                }
            }
        }
        else
        {
            // Tensor multiplication path
            torch::Tensor other;
            
            if (offset >= Size)
            {
                // Create a compatible tensor if no more data
                other = torch::ones_like(input);
            }
            else
            {
                // Parse second tensor
                try
                {
                    other = fuzzer_utils::createTensor(Data, Size, offset);
                }
                catch (const std::exception &e)
                {
                    // If parsing fails, create a compatible tensor
                    other = torch::randn_like(input);
                }
            }
            
            // Test broadcasting scenarios
            if (offset < Size && Data[offset++] % 4 == 0)
            {
                // Try to reshape for broadcasting test
                auto input_numel = input.numel();
                auto other_numel = other.numel();
                
                if (input_numel > 1 && other_numel > 1)
                {
                    // Try various broadcasting scenarios
                    uint8_t broadcast_type = (offset < Size) ? Data[offset++] % 5 : 0;
                    
                    try
                    {
                        switch (broadcast_type)
                        {
                        case 0:
                            // Make other a scalar tensor
                            other = other.reshape({1});
                            break;
                        case 1:
                            // Make other a column vector if input is 2D
                            if (input.dim() >= 2)
                            {
                                other = other.reshape({-1, 1});
                            }
                            break;
                        case 2:
                            // Make other a row vector if input is 2D
                            if (input.dim() >= 2)
                            {
                                other = other.reshape({1, -1});
                            }
                            break;
                        case 3:
                            // Try to match last dimension
                            if (input.dim() > 0)
                            {
                                auto last_dim = input.size(-1);
                                if (other.numel() % last_dim == 0)
                                {
                                    other = other.reshape({-1, last_dim});
                                }
                            }
                            break;
                        default:
                            // Keep original shape
                            break;
                        }
                    }
                    catch (const c10::Error &e)
                    {
                        // Ignore reshape errors, use original tensor
                    }
                }
            }
            
            // Perform multiplication
            if (use_out_tensor && offset < Size)
            {
                try
                {
                    // Try to create output tensor
                    // Determine output shape based on broadcasting rules
                    auto broadcast_shape = torch::broadcast_shapes(input.sizes(), other.sizes());
                    out_tensor = torch::empty(broadcast_shape, input.options());
                    torch::mul_out(out_tensor, input, other);
                    result = out_tensor;
                }
                catch (const c10::Error &e)
                {
                    // Fall back to regular multiplication if out fails
                    try
                    {
                        result = torch::mul(input, other);
                    }
                    catch (const c10::Error &e2)
                    {
                        // If multiplication fails (incompatible shapes), try element-wise with reshape
                        if (input.numel() == other.numel())
                        {
                            result = torch::mul(input.flatten(), other.flatten()).reshape(input.sizes());
                        }
                        else
                        {
                            // Last resort: multiply with broadcasted scalar
                            result = torch::mul(input, other.flatten()[0]);
                        }
                    }
                }
            }
            else
            {
                try
                {
                    result = torch::mul(input, other);
                }
                catch (const c10::Error &e)
                {
                    // Handle broadcasting errors
                    if (input.numel() == other.numel())
                    {
                        // Try flattened multiplication
                        result = torch::mul(input.flatten(), other.flatten()).reshape(input.sizes());
                    }
                    else
                    {
                        // Use first element of other as scalar
                        result = torch::mul(input, other.flatten()[0]);
                    }
                }
            }
        }
        
        // Additional operations to increase coverage
        if (offset < Size)
        {
            uint8_t extra_ops = Data[offset++];
            
            // Test in-place multiplication
            if (extra_ops % 3 == 0)
            {
                try
                {
                    torch::Tensor temp = input.clone();
                    if (use_scalar)
                    {
                        temp.mul_(2.0);
                    }
                    else
                    {
                        // Create compatible tensor for in-place op
                        torch::Tensor other_inplace = torch::ones_like(temp);
                        temp.mul_(other_inplace);
                    }
                }
                catch (const c10::Error &e)
                {
                    // Ignore in-place operation errors
                }
            }
            
            // Test multiplication with special values
            if (extra_ops % 5 == 0)
            {
                try
                {
                    torch::Tensor special = torch::mul(input, 0.0); // Multiply by zero
                    special = torch::mul(input, -1.0); // Multiply by negative one
                    special = torch::mul(input, std::numeric_limits<float>::infinity()); // Multiply by inf
                    special = torch::mul(input, std::numeric_limits<float>::quiet_NaN()); // Multiply by NaN
                }
                catch (const c10::Error &e)
                {
                    // Ignore errors with special values
                }
            }
        }
        
        // Validate result
        if (result.defined())
        {
            // Check for NaN or Inf in result
            bool has_nan = torch::any(torch::isnan(result)).item<bool>();
            bool has_inf = torch::any(torch::isinf(result)).item<bool>();
            
            // These are valid outcomes, just note them
            if (has_nan || has_inf)
            {
                // Valid mathematical results, continue
            }
            
            // Verify shape properties
            if (use_scalar)
            {
                // Result should have same shape as input for scalar multiplication
                if (result.sizes() != input.sizes())
                {
                    std::cerr << "Warning: Scalar multiplication changed tensor shape" << std::endl;
                }
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
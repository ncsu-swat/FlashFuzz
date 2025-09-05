#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful test cases
        if (Size < 4) {
            return 0;
        }

        // Parse operation mode from first byte
        uint8_t mode = Data[offset++];
        bool use_scalar_divisor = (mode & 0x01);
        bool use_out_tensor = (mode & 0x02);
        bool test_broadcasting = (mode & 0x04);
        bool test_inplace = (mode & 0x08);
        
        // Create dividend tensor
        torch::Tensor dividend = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Prepare divisor (tensor or scalar)
        torch::Tensor divisor;
        double scalar_divisor = 1.0;
        
        if (use_scalar_divisor && offset < Size) {
            // Use scalar divisor - parse from remaining bytes
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_divisor, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Handle special cases
                if (std::isnan(scalar_divisor) || std::isinf(scalar_divisor)) {
                    scalar_divisor = 1.0; // Normalize to avoid undefined behavior
                }
                if (scalar_divisor == 0.0) {
                    scalar_divisor = 0.0; // Keep zero to test division by zero
                }
            } else if (offset < Size) {
                // Use single byte as scalar
                scalar_divisor = static_cast<double>(Data[offset++]) / 128.0 - 1.0;
            }
        } else {
            // Create divisor tensor
            if (test_broadcasting && offset < Size) {
                // Create a tensor with different shape for broadcasting tests
                uint8_t broadcast_mode = Data[offset++];
                
                if (broadcast_mode % 3 == 0) {
                    // Scalar tensor
                    divisor = torch::ones({}, dividend.options());
                } else if (broadcast_mode % 3 == 1 && dividend.dim() > 0) {
                    // Single dimension matching last dim
                    auto shape = dividend.sizes();
                    divisor = fuzzer_utils::createTensor(Data, Size, offset);
                    if (shape.size() > 0) {
                        divisor = divisor.reshape({shape[shape.size()-1]});
                    }
                } else {
                    // Same shape or compatible broadcast shape
                    divisor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Try to make shapes broadcast-compatible
                    if (dividend.dim() > 0 && divisor.dim() > 0 && dividend.dim() != divisor.dim()) {
                        auto min_dim = std::min(dividend.dim(), divisor.dim());
                        std::vector<int64_t> new_shape;
                        for (int i = 0; i < min_dim; ++i) {
                            new_shape.push_back(1);
                        }
                        if (divisor.numel() > 0) {
                            divisor = divisor.reshape(new_shape);
                        }
                    }
                }
            } else {
                divisor = fuzzer_utils::createTensor(Data, Size, offset);
            }
            
            // Ensure divisor is same dtype as dividend for tensor operations
            if (divisor.defined() && divisor.dtype() != dividend.dtype()) {
                divisor = divisor.to(dividend.dtype());
            }
        }
        
        // Test different fmod variants
        torch::Tensor result;
        
        if (test_inplace && !use_scalar_divisor && divisor.defined()) {
            // Test in-place operation (fmod_)
            torch::Tensor temp = dividend.clone();
            
            // Ensure shapes are compatible for in-place
            if (temp.sizes() == divisor.sizes() || torch::are_expandable(temp.sizes(), divisor.sizes())) {
                temp.fmod_(divisor);
                result = temp;
            } else {
                // Fall back to regular fmod if shapes incompatible
                result = torch::fmod(dividend, divisor);
            }
        } else if (use_out_tensor) {
            // Test out-variant
            torch::Tensor out_tensor;
            
            if (use_scalar_divisor) {
                out_tensor = torch::empty_like(dividend);
                result = torch::fmod_out(out_tensor, dividend, scalar_divisor);
            } else if (divisor.defined()) {
                // Determine output shape for broadcasting
                auto output_shape = dividend.sizes().vec();
                if (divisor.dim() > 0) {
                    try {
                        auto broadcast_shape = torch::broadcast_shapes(dividend.sizes(), divisor.sizes());
                        output_shape = broadcast_shape;
                    } catch (...) {
                        // If broadcast fails, use dividend shape
                    }
                }
                out_tensor = torch::empty(output_shape, dividend.options());
                result = torch::fmod_out(out_tensor, dividend, divisor);
            } else {
                result = torch::fmod(dividend, 1.0);
            }
        } else {
            // Regular fmod operation
            if (use_scalar_divisor) {
                result = torch::fmod(dividend, scalar_divisor);
            } else if (divisor.defined()) {
                result = torch::fmod(dividend, divisor);
            } else {
                result = torch::fmod(dividend, 1.0);
            }
        }
        
        // Additional edge case testing
        if (offset < Size) {
            uint8_t edge_test = Data[offset++];
            
            if (edge_test & 0x01) {
                // Test with zero divisor
                auto zero_div = torch::zeros_like(dividend);
                auto zero_result = torch::fmod(dividend, zero_div);
                // Check for NaN/Inf in result
                auto has_nan = torch::isnan(zero_result).any().item<bool>();
                auto has_inf = torch::isinf(zero_result).any().item<bool>();
                (void)has_nan; (void)has_inf; // Suppress unused warnings
            }
            
            if (edge_test & 0x02) {
                // Test with negative values
                auto neg_dividend = -dividend;
                auto neg_result = use_scalar_divisor ? 
                    torch::fmod(neg_dividend, scalar_divisor) :
                    (divisor.defined() ? torch::fmod(neg_dividend, divisor) : torch::fmod(neg_dividend, 1.0));
                (void)neg_result;
            }
            
            if (edge_test & 0x04) {
                // Test with special values (if floating point)
                if (dividend.is_floating_point()) {
                    auto special_tensor = dividend.clone();
                    if (special_tensor.numel() > 0) {
                        special_tensor.view(-1)[0] = std::numeric_limits<float>::infinity();
                        if (special_tensor.numel() > 1) {
                            special_tensor.view(-1)[1] = -std::numeric_limits<float>::infinity();
                        }
                        if (special_tensor.numel() > 2) {
                            special_tensor.view(-1)[2] = std::numeric_limits<float>::quiet_NaN();
                        }
                        auto special_result = use_scalar_divisor ?
                            torch::fmod(special_tensor, scalar_divisor) :
                            (divisor.defined() ? torch::fmod(special_tensor, divisor) : torch::fmod(special_tensor, 1.0));
                        (void)special_result;
                    }
                }
            }
            
            if (edge_test & 0x08 && divisor.defined()) {
                // Test with mixed signs
                auto mixed_divisor = divisor.clone();
                if (mixed_divisor.numel() > 0) {
                    // Make some elements negative
                    auto mask = torch::randn_like(mixed_divisor) > 0;
                    mixed_divisor = torch::where(mask, mixed_divisor, -mixed_divisor);
                    auto mixed_result = torch::fmod(dividend, mixed_divisor);
                    (void)mixed_result;
                }
            }
        }
        
        // Validate result properties
        if (result.defined()) {
            // Check basic properties
            bool is_finite = torch::isfinite(result).all().item<bool>();
            (void)is_finite;
            
            // For integer types, result should be in range [0, |divisor|) or (-|divisor|, 0]
            // For floating point, similar but with sign considerations
            
            // Access some elements to ensure tensor is materialized
            if (result.numel() > 0) {
                auto first_elem = result.view(-1)[0];
                (void)first_elem;
                
                if (result.numel() > 1) {
                    auto last_elem = result.view(-1)[result.numel() - 1];
                    (void)last_elem;
                }
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected in fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
}
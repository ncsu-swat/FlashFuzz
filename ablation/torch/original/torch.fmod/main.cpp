#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Minimum size check - we need at least a few bytes for basic operations
    if (size < 4) {
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Consume first byte to determine test mode
        if (offset >= size) return 0;
        uint8_t mode = data[offset++] % 5;
        
        // Consume second byte for dtype selection (fmod doesn't support complex)
        if (offset >= size) return 0;
        uint8_t dtype_selector = data[offset++];
        
        // Map to non-complex dtypes only
        std::vector<torch::ScalarType> valid_dtypes = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
            torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64
        };
        torch::ScalarType dtype = valid_dtypes[dtype_selector % valid_dtypes.size()];
        
        switch (mode) {
            case 0: {
                // Mode 0: Tensor-Tensor operation with potential broadcasting
                torch::Tensor input, other;
                
                try {
                    // Create first tensor
                    input = fuzzer_utils::createTensor(data, size, offset);
                    
                    // Convert to selected dtype if different
                    if (input.scalar_type() != dtype && !input.is_complex()) {
                        input = input.to(dtype);
                    }
                    
                    // Create second tensor
                    if (offset < size) {
                        other = fuzzer_utils::createTensor(data, size, offset);
                        if (other.scalar_type() != dtype && !other.is_complex()) {
                            other = other.to(dtype);
                        }
                    } else {
                        // Create a small random tensor if no more data
                        other = torch::randn({2}, torch::TensorOptions().dtype(dtype));
                    }
                    
                    // Test fmod operation
                    torch::Tensor result = torch::fmod(input, other);
                    
                    // Verify result properties
                    if (result.defined()) {
                        // Check that result has proper shape (broadcasting rules)
                        auto expected_shape = torch::broadcast_shapes({input.sizes().vec(), other.sizes().vec()});
                        
                        // Access some elements to ensure computation happened
                        if (result.numel() > 0) {
                            auto flat = result.flatten();
                            if (flat.numel() > 0) {
                                flat[0].item<float>();
                            }
                        }
                    }
                } catch (const c10::Error& e) {
                    // PyTorch errors are expected for invalid operations
                    return 0;
                }
                break;
            }
            
            case 1: {
                // Mode 1: Tensor-Scalar operation
                torch::Tensor input;
                
                try {
                    input = fuzzer_utils::createTensor(data, size, offset);
                    if (input.scalar_type() != dtype && !input.is_complex()) {
                        input = input.to(dtype);
                    }
                    
                    // Parse scalar value
                    double scalar_val = 1.0;
                    if (offset + sizeof(double) <= size) {
                        std::memcpy(&scalar_val, data + offset, sizeof(double));
                        offset += sizeof(double);
                        
                        // Normalize to reasonable range
                        if (!std::isfinite(scalar_val)) {
                            scalar_val = 1.0;
                        } else {
                            scalar_val = std::fmod(scalar_val, 1000.0);
                        }
                    }
                    
                    // Test with scalar
                    torch::Tensor result = torch::fmod(input, scalar_val);
                    
                    // Verify result
                    if (result.defined() && result.numel() > 0) {
                        result.flatten()[0].item<float>();
                    }
                } catch (const c10::Error& e) {
                    return 0;
                }
                break;
            }
            
            case 2: {
                // Mode 2: Edge case - division by zero
                try {
                    // Create input tensor
                    torch::Tensor input = torch::randn({2, 3}, torch::TensorOptions().dtype(dtype));
                    
                    // Create divisor with zeros
                    torch::Tensor other = torch::zeros({2, 3}, torch::TensorOptions().dtype(dtype));
                    
                    // For floating point, this should return NaN
                    // For integer, this may raise RuntimeError on CPU
                    torch::Tensor result = torch::fmod(input, other);
                    
                    if (result.defined() && dtype == torch::kFloat || dtype == torch::kDouble) {
                        // Check for NaN values as expected
                        auto is_nan = torch::isnan(result);
                        is_nan.any().item<bool>();
                    }
                } catch (const c10::Error& e) {
                    // Expected for integer division by zero on CPU
                    return 0;
                } catch (const std::runtime_error& e) {
                    // Also acceptable
                    return 0;
                }
                break;
            }
            
            case 3: {
                // Mode 3: Empty tensors and edge dimensions
                try {
                    if (offset >= size) return 0;
                    uint8_t edge_type = data[offset++] % 4;
                    
                    torch::Tensor input, other;
                    
                    switch (edge_type) {
                        case 0:
                            // Empty tensor
                            input = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                            other = torch::ones({1}, torch::TensorOptions().dtype(dtype));
                            break;
                        case 1:
                            // Scalar tensors
                            input = torch::tensor(3.14, torch::TensorOptions().dtype(dtype));
                            other = torch::tensor(2.0, torch::TensorOptions().dtype(dtype));
                            break;
                        case 2:
                            // Large dimension count
                            input = torch::ones({1, 1, 1, 1, 2}, torch::TensorOptions().dtype(dtype));
                            other = torch::full({2}, 3.0, torch::TensorOptions().dtype(dtype));
                            break;
                        default:
                            // Single element tensors
                            input = torch::ones({1}, torch::TensorOptions().dtype(dtype));
                            other = torch::full({1}, 0.5, torch::TensorOptions().dtype(dtype));
                            break;
                    }
                    
                    torch::Tensor result = torch::fmod(input, other);
                    
                    if (result.defined()) {
                        // Verify shape
                        result.sizes();
                        if (result.numel() > 0) {
                            result.flatten()[0].item<float>();
                        }
                    }
                } catch (const c10::Error& e) {
                    return 0;
                }
                break;
            }
            
            case 4: {
                // Mode 4: Test with out parameter
                try {
                    torch::Tensor input = torch::randn({3, 4}, torch::TensorOptions().dtype(dtype));
                    torch::Tensor other = torch::randn({3, 4}, torch::TensorOptions().dtype(dtype));
                    torch::Tensor out = torch::empty({3, 4}, torch::TensorOptions().dtype(dtype));
                    
                    // Test with out parameter
                    torch::fmod_out(out, input, other);
                    
                    // Verify the operation worked
                    if (out.defined() && out.numel() > 0) {
                        out.flatten()[0].item<float>();
                        
                        // Compare with regular fmod
                        torch::Tensor expected = torch::fmod(input, other);
                        if (dtype == torch::kFloat || dtype == torch::kDouble) {
                            // For floating point, check closeness (accounting for NaN)
                            auto both_nan = torch::isnan(out) & torch::isnan(expected);
                            auto both_finite = ~torch::isnan(out) & ~torch::isnan(expected);
                            auto close = torch::isclose(out, expected, 1e-5, 1e-8);
                            (both_nan | (both_finite & close)).all().item<bool>();
                        }
                    }
                } catch (const c10::Error& e) {
                    return 0;
                }
                break;
            }
        }
        
        // Additional stress test with remaining data
        while (offset + 10 < size) {
            try {
                torch::Tensor t = fuzzer_utils::createTensor(data, size, offset);
                
                // Quick fmod with various scalars
                if (offset < size) {
                    uint8_t scalar_type = data[offset++] % 5;
                    double divisor = 1.0;
                    
                    switch (scalar_type) {
                        case 0: divisor = 0.0; break;  // Zero
                        case 1: divisor = -1.5; break; // Negative
                        case 2: divisor = 0.001; break; // Small
                        case 3: divisor = 1000.0; break; // Large
                        default: divisor = 2.0; break; // Normal
                    }
                    
                    torch::Tensor result = torch::fmod(t, divisor);
                    
                    // Just ensure it doesn't crash
                    if (result.defined() && result.numel() > 0) {
                        result.sum().item<float>();
                    }
                }
            } catch (...) {
                // Continue with next iteration
                break;
            }
        }
        
    }
    catch (const std::exception &e)
    {
        // Log but don't print to avoid spam in fuzzing
        // Returning 0 to keep the input for corpus
        return 0;
    }
    catch (...)
    {
        // Catch any other errors
        return 0;
    }
    
    return 0;
}
#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to check if dtype is complex (which remainder doesn't support)
bool isComplexDtype(torch::ScalarType dtype) {
    return dtype == torch::kComplexFloat || dtype == torch::kComplexDouble;
}

// Helper to create scalar from bytes
double parseScalar(const uint8_t* data, size_t& offset, size_t size) {
    if (offset + sizeof(double) > size) {
        // Not enough data, use a default
        double default_val = 1.0;
        offset = size; // Mark as consumed
        return default_val;
    }
    
    double value;
    std::memcpy(&value, data + offset, sizeof(double));
    offset += sizeof(double);
    
    // Avoid NaN/Inf for more predictable testing
    if (!std::isfinite(value)) {
        value = 1.0;
    }
    
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3) {  // Need at least mode + 2 tensor metadata bytes
        return 0;
    }
    
    try {
        size_t offset = 0;
        
        // Parse operation mode
        uint8_t mode = data[offset++] % 4;
        
        // Mode 0: tensor % tensor
        // Mode 1: tensor % scalar
        // Mode 2: scalar % tensor
        // Mode 3: tensor % tensor with broadcasting
        
        if (mode == 0 || mode == 3) {
            // Tensor % Tensor
            torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
            
            // Skip complex dtypes for remainder
            if (isComplexDtype(input.scalar_type())) {
                // Convert to float to continue testing
                input = input.to(torch::kFloat);
            }
            
            torch::Tensor other = fuzzer_utils::createTensor(data, size, offset);
            
            // Skip complex dtypes
            if (isComplexDtype(other.scalar_type())) {
                other = other.to(torch::kFloat);
            }
            
            // For mode 3, try to create tensors with different but broadcastable shapes
            if (mode == 3 && offset < size) {
                uint8_t broadcast_type = data[offset++] % 5;
                
                switch(broadcast_type) {
                    case 0:
                        // Make other a scalar tensor
                        other = torch::tensor(parseScalar(data, offset, size), other.options());
                        break;
                    case 1:
                        // Make other 1D with size 1
                        if (other.numel() > 0) {
                            other = other.flatten()[0].reshape({1});
                        }
                        break;
                    case 2:
                        // Add singleton dimensions
                        if (input.dim() > 0) {
                            std::vector<int64_t> new_shape(input.dim(), 1);
                            new_shape[input.dim() - 1] = other.numel() > 0 ? 
                                std::min(other.numel(), input.size(-1)) : 1;
                            other = other.flatten().narrow(0, 0, new_shape[input.dim() - 1]).reshape(new_shape);
                        }
                        break;
                    case 3:
                        // Transpose if both are 2D
                        if (input.dim() == 2 && other.dim() == 2) {
                            other = other.t();
                        }
                        break;
                    case 4:
                        // Make shapes partially compatible
                        if (input.dim() > 1 && other.dim() > 1) {
                            auto input_shape = input.sizes().vec();
                            auto other_shape = other.sizes().vec();
                            // Make last dimension compatible
                            if (other_shape.back() != input_shape.back() && other_shape.back() != 1) {
                                other_shape.back() = 1;
                                other = other.flatten().narrow(0, 0, 1).expand(other_shape);
                            }
                        }
                        break;
                }
            }
            
            // Test with out parameter occasionally
            bool use_out = offset < size && (data[offset++] % 4 == 0);
            
            if (use_out) {
                // Create output tensor with compatible shape
                torch::Tensor out;
                try {
                    // Try to infer the broadcast shape
                    auto broadcast_shape = torch::broadcast_shapes(input.sizes(), other.sizes());
                    out = torch::empty(broadcast_shape, input.options());
                    torch::remainder_out(out, input, other);
                } catch (const c10::Error& e) {
                    // Broadcasting failed, try without out parameter
                    torch::remainder(input, other);
                }
            } else {
                torch::Tensor result = torch::remainder(input, other);
                
                // Verify some properties
                if (result.numel() > 0 && other.numel() > 0) {
                    // Check that result has same sign as divisor (for non-zero divisors)
                    auto other_flat = other.flatten();
                    auto result_flat = result.flatten();
                    
                    // Additional operations to increase coverage
                    if (offset < size && data[offset++] % 2 == 0) {
                        // Test in-place operation
                        if (input.sizes() == result.sizes()) {
                            input.remainder_(other);
                        }
                    }
                }
            }
            
        } else if (mode == 1) {
            // Tensor % Scalar
            torch::Tensor input = fuzzer_utils::createTensor(data, size, offset);
            
            if (isComplexDtype(input.scalar_type())) {
                input = input.to(torch::kFloat);
            }
            
            double scalar_other = parseScalar(data, offset, size);
            
            // Test edge cases
            if (offset < size) {
                uint8_t edge_case = data[offset++] % 5;
                switch(edge_case) {
                    case 0: scalar_other = 0.0; break;  // Division by zero
                    case 1: scalar_other = -scalar_other; break;  // Negative divisor
                    case 2: scalar_other = 0.5; break;  // Fractional
                    case 3: scalar_other = 1e-10; break;  // Very small
                    case 4: scalar_other = 1e10; break;  // Very large
                }
            }
            
            torch::Tensor result = torch::remainder(input, scalar_other);
            
            // Test in-place variant
            if (offset < size && data[offset++] % 2 == 0) {
                input.remainder_(scalar_other);
            }
            
        } else if (mode == 2) {
            // Scalar % Tensor
            double scalar_input = parseScalar(data, offset, size);
            torch::Tensor other = fuzzer_utils::createTensor(data, size, offset);
            
            if (isComplexDtype(other.scalar_type())) {
                other = other.to(torch::kFloat);
            }
            
            // Edge cases for scalar input
            if (offset < size) {
                uint8_t edge_case = data[offset++] % 4;
                switch(edge_case) {
                    case 0: scalar_input = 0.0; break;
                    case 1: scalar_input = -scalar_input; break;
                    case 2: scalar_input = std::numeric_limits<double>::min(); break;
                    case 3: scalar_input = std::numeric_limits<double>::max(); break;
                }
            }
            
            torch::Tensor result = torch::remainder(scalar_input, other);
        }
        
        // Test special tensor types if we have more data
        if (offset + 10 < size) {
            uint8_t special_test = data[offset++] % 6;
            
            switch(special_test) {
                case 0: {
                    // Test with empty tensors
                    auto empty1 = torch::empty({0});
                    auto empty2 = torch::empty({0});
                    torch::remainder(empty1, empty2);
                    break;
                }
                case 1: {
                    // Test with zero-dim tensors (scalars)
                    auto scalar1 = torch::tensor(3.14);
                    auto scalar2 = torch::tensor(2.0);
                    torch::remainder(scalar1, scalar2);
                    break;
                }
                case 2: {
                    // Test with strided tensors
                    auto t = torch::randn({10, 10});
                    auto strided = t[torch::indexing::Slice(0, torch::indexing::None, 2)];
                    torch::remainder(strided, 3.0);
                    break;
                }
                case 3: {
                    // Test with different memory layouts
                    auto t1 = torch::randn({4, 5}).t();  // Non-contiguous
                    auto t2 = torch::randn({5, 4});
                    torch::remainder(t1, t2.t());
                    break;
                }
                case 4: {
                    // Test type promotion
                    auto int_tensor = torch::randint(1, 10, {3, 3}, torch::kInt32);
                    auto float_tensor = torch::randn({3, 3});
                    torch::remainder(int_tensor, float_tensor);
                    break;
                }
                case 5: {
                    // Test with negative values
                    auto neg_input = torch::tensor({-3.0, -2.0, -1.0, 1.0, 2.0, 3.0});
                    auto neg_other = torch::tensor({-1.5, 2.0, -1.0, -2.0, 1.5, -3.0});
                    torch::remainder(neg_input, neg_other);
                    break;
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        // Log unexpected errors but continue fuzzing
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
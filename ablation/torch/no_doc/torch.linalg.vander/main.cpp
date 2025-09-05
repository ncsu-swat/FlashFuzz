#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and N parameter
        if (Size < 4) {
            return 0;
        }

        // Create input tensor - vander expects 1-D input
        torch::Tensor x;
        try {
            x = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with minimal valid tensor
            x = torch::randn({3});
        }
        
        // Ensure tensor is 1-D for vander (reshape if needed)
        if (x.dim() != 1) {
            // Flatten to 1-D
            x = x.flatten();
            
            // Handle edge case of empty tensor
            if (x.numel() == 0) {
                x = torch::randn({1});
            }
        }
        
        // Parse N (number of columns) if data available
        c10::optional<int64_t> N;
        if (offset < Size) {
            // Read N value from remaining bytes
            uint8_t n_selector = Data[offset++];
            
            // Map to reasonable range [0, 100] with some special cases
            if (n_selector < 10) {
                // Test edge cases: 0, 1, 2
                N = static_cast<int64_t>(n_selector % 3);
            } else if (n_selector < 20) {
                // Test negative values (should trigger errors)
                N = -static_cast<int64_t>(n_selector % 10) - 1;
            } else if (n_selector < 30) {
                // Test nullptr case (no N specified)
                N = c10::nullopt;
            } else {
                // Normal range [1, 100]
                N = static_cast<int64_t>(n_selector % 100) + 1;
            }
        }
        
        // Parse increasing flag if data available
        bool increasing = true;
        if (offset < Size) {
            increasing = (Data[offset++] % 2) == 0;
        }
        
        // Test different device/dtype combinations if data available
        if (offset < Size && (Data[offset] % 10) == 0) {
            offset++;
            // Try converting to different dtypes
            auto dtype_selector = offset < Size ? Data[offset++] : 0;
            try {
                switch (dtype_selector % 6) {
                    case 0: x = x.to(torch::kFloat32); break;
                    case 1: x = x.to(torch::kFloat64); break;
                    case 2: x = x.to(torch::kComplex64); break;
                    case 3: x = x.to(torch::kComplex128); break;
                    case 4: x = x.to(torch::kInt32); break;
                    case 5: x = x.to(torch::kInt64); break;
                }
            } catch (...) {
                // Ignore conversion errors
            }
        }
        
        // Test with requires_grad if data available
        if (offset < Size && (Data[offset] % 5) == 0) {
            offset++;
            if (x.dtype().isFloatingPoint() || x.dtype().isComplex()) {
                x = x.requires_grad_(true);
            }
        }
        
        // Call torch::linalg::vander with different parameter combinations
        torch::Tensor result;
        
        try {
            if (N.has_value()) {
                // Call with explicit N
                result = torch::linalg::vander(x, N.value(), increasing);
                
                // Validate output shape
                if (result.size(0) != x.size(0)) {
                    std::cerr << "Unexpected row count in result" << std::endl;
                }
                if (N.value() >= 0 && result.size(1) != N.value()) {
                    std::cerr << "Unexpected column count in result" << std::endl;
                }
            } else {
                // Call without N (uses x.size(0) as default)
                result = torch::linalg::vander(x);
                
                // Validate square matrix output
                if (result.size(0) != x.size(0) || result.size(1) != x.size(0)) {
                    std::cerr << "Result should be square when N not specified" << std::endl;
                }
            }
            
            // Additional validation and edge case testing
            if (result.defined()) {
                // Check first/last column based on increasing flag
                if (result.size(1) > 0) {
                    auto first_col = increasing ? result.select(1, 0) : result.select(1, result.size(1) - 1);
                    if (!torch::allclose(first_col, torch::ones_like(first_col), 1e-5, 1e-8)) {
                        // First column (or last if decreasing) should be all ones
                    }
                }
                
                // Test backward pass if applicable
                if (result.requires_grad()) {
                    try {
                        auto loss = result.sum();
                        loss.backward();
                    } catch (...) {
                        // Ignore backward errors
                    }
                }
                
                // Test with various tensor operations
                if (offset < Size && (Data[offset] % 3) == 0) {
                    try {
                        auto transposed = result.t();
                        auto matmul = torch::matmul(result, transposed);
                        auto det = torch::linalg::det(result);
                    } catch (...) {
                        // Ignore operation errors
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (expected for invalid inputs)
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors (expected for some edge cases)
            return 0;
        }
        
        // Test edge cases with special input values
        if (offset < Size && (Data[offset] % 7) == 0) {
            try {
                // Test with special values
                auto special_x = torch::tensor({0.0, 1.0, -1.0, 2.0, 0.5});
                auto special_result = torch::linalg::vander(special_x);
                
                // Test with infinity/nan if floating point
                if (x.dtype().isFloatingPoint()) {
                    auto inf_x = torch::tensor({std::numeric_limits<float>::infinity(), 1.0, -1.0});
                    auto inf_result = torch::linalg::vander(inf_x, 3);
                    
                    auto nan_x = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 1.0, 2.0});
                    auto nan_result = torch::linalg::vander(nan_x);
                }
            } catch (...) {
                // Ignore errors from special values
            }
        }
        
        // Test with empty tensor
        if (offset < Size && (Data[offset] % 11) == 0) {
            try {
                auto empty_x = torch::empty({0});
                auto empty_result = torch::linalg::vander(empty_x);
                if (empty_result.size(0) != 0 || empty_result.size(1) != 0) {
                    std::cerr << "Empty input should produce empty output" << std::endl;
                }
            } catch (...) {
                // Some implementations might not handle empty tensors
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
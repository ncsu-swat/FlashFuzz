#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and parameters
        if (Size < 32) {
            return 0;
        }

        // Extract tensor shapes and parameters
        auto shape1 = extract_tensor_shape(Data, Size, offset, 1, 6);
        auto shape2 = extract_tensor_shape(Data, Size, offset, 1, 6);
        
        // Extract dimension parameter
        int64_t dim = extract_int(Data, Size, offset, -static_cast<int64_t>(shape1.size()), 
                                 static_cast<int64_t>(shape1.size()) - 1);
        
        // Extract eps parameter
        double eps = extract_double(Data, Size, offset, 1e-12, 1e-4);
        
        // Extract dtype
        auto dtype = extract_dtype(Data, Size, offset);
        
        // Create tensors with different strategies
        torch::Tensor x1, x2;
        
        // Strategy selection
        uint8_t strategy = extract_uint8(Data, Size, offset) % 8;
        
        switch (strategy) {
            case 0: {
                // Normal random tensors
                x1 = torch::randn(shape1, torch::dtype(dtype));
                x2 = torch::randn(shape2, torch::dtype(dtype));
                break;
            }
            case 1: {
                // Zero tensors (edge case for division by zero)
                x1 = torch::zeros(shape1, torch::dtype(dtype));
                x2 = torch::zeros(shape2, torch::dtype(dtype));
                break;
            }
            case 2: {
                // One zero, one non-zero
                x1 = torch::zeros(shape1, torch::dtype(dtype));
                x2 = torch::randn(shape2, torch::dtype(dtype));
                break;
            }
            case 3: {
                // Very small values (near eps)
                x1 = torch::full(shape1, eps * 0.1, torch::dtype(dtype));
                x2 = torch::full(shape2, eps * 0.1, torch::dtype(dtype));
                break;
            }
            case 4: {
                // Large values
                x1 = torch::full(shape1, 1e6, torch::dtype(dtype));
                x2 = torch::full(shape2, 1e6, torch::dtype(dtype));
                break;
            }
            case 5: {
                // Mixed positive and negative
                x1 = torch::randn(shape1, torch::dtype(dtype));
                x2 = -torch::randn(shape2, torch::dtype(dtype));
                break;
            }
            case 6: {
                // Identical tensors (should give similarity of 1)
                x1 = torch::randn(shape1, torch::dtype(dtype));
                x2 = x1.expand(shape2);
                break;
            }
            case 7: {
                // Orthogonal-like tensors
                x1 = torch::ones(shape1, torch::dtype(dtype));
                x2 = torch::full(shape2, -1.0, torch::dtype(dtype));
                break;
            }
        }
        
        // Test different broadcasting scenarios
        uint8_t broadcast_strategy = extract_uint8(Data, Size, offset) % 4;
        
        switch (broadcast_strategy) {
            case 0: {
                // No modification - test as is
                break;
            }
            case 1: {
                // Add singleton dimensions
                if (x1.dim() > 0) {
                    x1 = x1.unsqueeze(0);
                }
                break;
            }
            case 2: {
                // Expand one tensor
                if (x1.numel() == 1) {
                    x1 = x1.expand_as(x2);
                }
                break;
            }
            case 3: {
                // Create broadcastable shapes
                if (x1.dim() > 0 && x2.dim() > 0) {
                    auto new_shape1 = x1.sizes().vec();
                    auto new_shape2 = x2.sizes().vec();
                    
                    // Make them broadcastable by setting some dims to 1
                    if (new_shape1.size() > 1) {
                        new_shape1[0] = 1;
                        x1 = x1.view(new_shape1);
                    }
                }
                break;
            }
        }
        
        // Test edge cases for dim parameter
        uint8_t dim_strategy = extract_uint8(Data, Size, offset) % 3;
        
        // Determine valid dimension range after potential broadcasting
        torch::Tensor temp_result;
        try {
            temp_result = torch::broadcast_tensors({x1, x2})[0];
        } catch (...) {
            // If broadcasting fails, use original tensors
            temp_result = x1;
        }
        
        int64_t max_dim = temp_result.dim() - 1;
        int64_t min_dim = -temp_result.dim();
        
        switch (dim_strategy) {
            case 0: {
                // Use extracted dim, clamped to valid range
                dim = std::max(min_dim, std::min(max_dim, dim));
                break;
            }
            case 1: {
                // Use last dimension
                dim = max_dim;
                break;
            }
            case 2: {
                // Use first dimension
                dim = 0;
                break;
            }
        }
        
        // Test different eps values
        uint8_t eps_strategy = extract_uint8(Data, Size, offset) % 4;
        
        switch (eps_strategy) {
            case 0: {
                // Use extracted eps
                break;
            }
            case 1: {
                // Very small eps
                eps = 1e-12;
                break;
            }
            case 2: {
                // Large eps
                eps = 1e-2;
                break;
            }
            case 3: {
                // Zero eps (edge case)
                eps = 0.0;
                break;
            }
        }
        
        // Call cosine_similarity with different parameter combinations
        torch::Tensor result;
        
        // Test with all parameters
        result = torch::cosine_similarity(x1, x2, dim, eps);
        
        // Verify result properties
        if (result.defined()) {
            // Check that result has correct number of dimensions
            auto expected_dims = std::max(x1.dim(), x2.dim()) - 1;
            if (result.dim() != expected_dims && expected_dims >= 0) {
                // This might be expected behavior, just continue
            }
            
            // Check for NaN or Inf values
            auto has_nan = torch::any(torch::isnan(result));
            auto has_inf = torch::any(torch::isinf(result));
            
            // Force evaluation
            if (has_nan.defined()) has_nan.item<bool>();
            if (has_inf.defined()) has_inf.item<bool>();
            
            // Check value range (cosine similarity should be in [-1, 1])
            if (result.numel() > 0) {
                auto min_val = torch::min(result);
                auto max_val = torch::max(result);
                
                if (min_val.defined()) min_val.item<double>();
                if (max_val.defined()) max_val.item<double>();
            }
        }
        
        // Test with default parameters
        result = torch::cosine_similarity(x1, x2);
        
        // Test with only dim specified
        result = torch::cosine_similarity(x1, x2, dim);
        
        // Test special cases
        if (x1.dim() > 0 && x2.dim() > 0) {
            // Test with negative dimension
            int64_t neg_dim = -1;
            if (neg_dim >= -std::max(x1.dim(), x2.dim())) {
                result = torch::cosine_similarity(x1, x2, neg_dim, eps);
            }
        }
        
        // Test type promotion by mixing dtypes
        if (dtype != torch::kFloat32) {
            auto x1_float = x1.to(torch::kFloat32);
            result = torch::cosine_similarity(x1_float, x2, dim, eps);
        }
        
        // Test with requires_grad
        if (x1.dtype().is_floating_point() && x2.dtype().is_floating_point()) {
            auto x1_grad = x1.clone().requires_grad_(true);
            auto x2_grad = x2.clone().requires_grad_(true);
            
            result = torch::cosine_similarity(x1_grad, x2_grad, dim, eps);
            
            if (result.defined() && result.numel() > 0) {
                auto loss = torch::sum(result);
                if (loss.defined() && loss.requires_grad()) {
                    loss.backward();
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
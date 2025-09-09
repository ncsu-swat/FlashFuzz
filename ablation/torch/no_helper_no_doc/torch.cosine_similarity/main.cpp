#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensor creation and cosine similarity
        if (Size < 32) {
            return 0;
        }

        // Extract parameters for tensor creation
        auto shape_info = extract_tensor_shape(Data, Size, offset);
        if (shape_info.empty()) {
            return 0;
        }

        // Extract dimension parameter for cosine similarity
        int64_t dim = extract_int64_t(Data, Size, offset) % 4; // Limit to reasonable range
        if (dim < 0) dim = -dim; // Make positive for modulo operation
        dim = dim % std::max(1, (int)shape_info.size()); // Ensure valid dimension

        // Extract epsilon parameter
        double eps = extract_double(Data, Size, offset);
        // Clamp epsilon to reasonable range to avoid numerical issues
        eps = std::max(1e-12, std::min(1e-1, std::abs(eps)));

        // Create two input tensors for cosine similarity
        torch::Tensor x1, x2;
        
        // Try different tensor creation strategies based on remaining data
        if (offset + 8 < Size) {
            uint8_t strategy = Data[offset++];
            
            switch (strategy % 4) {
                case 0: {
                    // Create tensors with random values
                    x1 = create_tensor_from_data(Data, Size, offset, shape_info);
                    x2 = create_tensor_from_data(Data, Size, offset, shape_info);
                    break;
                }
                case 1: {
                    // Create tensors with specific patterns
                    x1 = torch::randn(shape_info);
                    x2 = torch::randn(shape_info);
                    break;
                }
                case 2: {
                    // Create tensors with zeros and ones
                    x1 = torch::zeros(shape_info);
                    x2 = torch::ones(shape_info);
                    break;
                }
                case 3: {
                    // Create identical tensors
                    x1 = create_tensor_from_data(Data, Size, offset, shape_info);
                    x2 = x1.clone();
                    break;
                }
            }
        } else {
            // Fallback: create simple tensors
            x1 = torch::randn(shape_info);
            x2 = torch::randn(shape_info);
        }

        // Ensure tensors are valid and have the same shape
        if (x1.numel() == 0 || x2.numel() == 0) {
            return 0;
        }

        // Make sure both tensors have the same shape
        x2 = x2.reshape(x1.sizes());

        // Test cosine similarity with different parameter combinations
        
        // Test 1: Basic cosine similarity with default parameters
        auto result1 = torch::cosine_similarity(x1, x2);
        
        // Test 2: Cosine similarity with specified dimension
        if (x1.dim() > 0) {
            auto result2 = torch::cosine_similarity(x1, x2, dim);
        }
        
        // Test 3: Cosine similarity with specified dimension and epsilon
        if (x1.dim() > 0) {
            auto result3 = torch::cosine_similarity(x1, x2, dim, eps);
        }

        // Test edge cases
        if (offset + 4 < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 6) {
                case 0: {
                    // Test with very small values
                    auto small_x1 = x1 * 1e-10;
                    auto small_x2 = x2 * 1e-10;
                    auto result_small = torch::cosine_similarity(small_x1, small_x2, dim, eps);
                    break;
                }
                case 1: {
                    // Test with very large values
                    auto large_x1 = x1 * 1e10;
                    auto large_x2 = x2 * 1e10;
                    auto result_large = torch::cosine_similarity(large_x1, large_x2, dim, eps);
                    break;
                }
                case 2: {
                    // Test with negative dimension
                    if (x1.dim() > 0) {
                        int64_t neg_dim = -1;
                        auto result_neg = torch::cosine_similarity(x1, x2, neg_dim, eps);
                    }
                    break;
                }
                case 3: {
                    // Test with zero tensors
                    auto zero_x1 = torch::zeros_like(x1);
                    auto zero_x2 = torch::zeros_like(x2);
                    auto result_zero = torch::cosine_similarity(zero_x1, zero_x2, dim, eps);
                    break;
                }
                case 4: {
                    // Test with orthogonal vectors (if possible)
                    if (x1.dim() >= 1 && x1.size(-1) >= 2) {
                        auto orth_x1 = torch::zeros_like(x1);
                        auto orth_x2 = torch::zeros_like(x2);
                        orth_x1.select(-1, 0).fill_(1.0);
                        orth_x2.select(-1, 1).fill_(1.0);
                        auto result_orth = torch::cosine_similarity(orth_x1, orth_x2, dim, eps);
                    }
                    break;
                }
                case 5: {
                    // Test with different dtypes if possible
                    if (x1.dtype() != torch::kFloat64) {
                        auto double_x1 = x1.to(torch::kFloat64);
                        auto double_x2 = x2.to(torch::kFloat64);
                        auto result_double = torch::cosine_similarity(double_x1, double_x2, dim, eps);
                    }
                    break;
                }
            }
        }

        // Test with different tensor shapes if we have enough data
        if (offset + 16 < Size) {
            // Create tensors with different but compatible shapes for broadcasting
            auto shape2 = extract_tensor_shape(Data, Size, offset);
            if (!shape2.empty()) {
                try {
                    auto x3 = create_tensor_from_data(Data, Size, offset, shape2);
                    if (x3.numel() > 0) {
                        // Try cosine similarity with potentially different shapes
                        auto result_broadcast = torch::cosine_similarity(x1, x3, dim, eps);
                    }
                } catch (...) {
                    // Ignore broadcasting errors
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
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 16) return 0;

        // Extract tensor dimensions
        int64_t batch_size = extract_int64_t(Data, Size, offset, 1, 10);
        int64_t seq_len = extract_int64_t(Data, Size, offset, 1, 100);
        int64_t hidden_dim = extract_int64_t(Data, Size, offset, 1, 50);
        
        // Extract k value (number of top elements to return)
        int64_t k = extract_int64_t(Data, Size, offset, 1, std::min(hidden_dim, (int64_t)20));
        
        // Extract dimension to operate on
        int64_t dim = extract_int64_t(Data, Size, offset, -3, 2); // -3 to 2 for 3D tensor
        
        // Extract boolean flags
        bool largest = extract_bool(Data, Size, offset);
        bool sorted = extract_bool(Data, Size, offset);
        
        // Create input tensor with various shapes and dtypes
        torch::Tensor input;
        int dtype_choice = extract_int(Data, Size, offset, 0, 3);
        
        switch (dtype_choice) {
            case 0:
                input = torch::randn({batch_size, seq_len, hidden_dim}, torch::kFloat32);
                break;
            case 1:
                input = torch::randn({batch_size, seq_len, hidden_dim}, torch::kFloat64);
                break;
            case 2:
                input = torch::randint(-100, 100, {batch_size, seq_len, hidden_dim}, torch::kInt32);
                break;
            case 3:
                input = torch::randint(-1000, 1000, {batch_size, seq_len, hidden_dim}, torch::kInt64);
                break;
        }
        
        // Add some edge case values
        if (extract_bool(Data, Size, offset)) {
            input[0][0][0] = std::numeric_limits<float>::infinity();
        }
        if (extract_bool(Data, Size, offset)) {
            input[0][0][1] = -std::numeric_limits<float>::infinity();
        }
        if (extract_bool(Data, Size, offset)) {
            input[0][0][2] = std::numeric_limits<float>::quiet_NaN();
        }
        
        // Test different tensor shapes
        int shape_choice = extract_int(Data, Size, offset, 0, 4);
        switch (shape_choice) {
            case 0: // 1D tensor
                input = input.view({-1});
                dim = 0;
                k = std::min(k, input.size(0));
                break;
            case 1: // 2D tensor
                input = input.view({batch_size, -1});
                dim = extract_int64_t(Data, Size, offset, -2, 1);
                k = std::min(k, input.size(dim < 0 ? dim + 2 : dim));
                break;
            case 2: // 3D tensor (keep as is)
                k = std::min(k, input.size(dim < 0 ? dim + 3 : dim));
                break;
            case 3: // 4D tensor
                input = input.view({batch_size, seq_len, hidden_dim/2, 2});
                dim = extract_int64_t(Data, Size, offset, -4, 3);
                k = std::min(k, input.size(dim < 0 ? dim + 4 : dim));
                break;
            case 4: // Empty tensor
                input = torch::empty({0});
                k = 0;
                dim = 0;
                break;
        }
        
        // Ensure k is valid
        if (input.numel() > 0) {
            int64_t actual_dim = dim < 0 ? dim + input.dim() : dim;
            if (actual_dim >= 0 && actual_dim < input.dim()) {
                k = std::min(k, input.size(actual_dim));
            }
        }
        k = std::max(k, (int64_t)0);
        
        // Test torch::topk with different parameter combinations
        auto result1 = torch::topk(input, k);
        auto values1 = std::get<0>(result1);
        auto indices1 = std::get<1>(result1);
        
        // Test with explicit parameters
        if (input.numel() > 0 && k > 0) {
            auto result2 = torch::topk(input, k, dim, largest, sorted);
            auto values2 = std::get<0>(result2);
            auto indices2 = std::get<1>(result2);
            
            // Verify output shapes
            if (values2.numel() > 0) {
                values2.sum(); // Force computation
                indices2.sum(); // Force computation
            }
        }
        
        // Test edge cases
        if (input.numel() > 0) {
            // k = 1
            auto result_k1 = torch::topk(input, 1, dim, largest, sorted);
            std::get<0>(result_k1).sum();
            std::get<1>(result_k1).sum();
            
            // Test with different dimensions if tensor has multiple dims
            if (input.dim() > 1) {
                for (int64_t test_dim = 0; test_dim < input.dim(); ++test_dim) {
                    int64_t test_k = std::min((int64_t)3, input.size(test_dim));
                    if (test_k > 0) {
                        auto result_dim = torch::topk(input, test_k, test_dim, largest, sorted);
                        std::get<0>(result_dim).sum();
                        std::get<1>(result_dim).sum();
                    }
                }
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input.numel() > 0 && input.dim() > 1) {
            auto transposed = input.transpose(0, 1);
            if (!transposed.is_contiguous() && k > 0) {
                auto result_nc = torch::topk(transposed, k, dim, largest, sorted);
                std::get<0>(result_nc).sum();
                std::get<1>(result_nc).sum();
            }
        }
        
        // Test with very small tensors
        if (extract_bool(Data, Size, offset)) {
            auto small_tensor = torch::randn({1});
            auto small_result = torch::topk(small_tensor, 1);
            std::get<0>(small_result).sum();
            std::get<1>(small_result).sum();
        }
        
        // Test with tensors containing special values
        if (extract_bool(Data, Size, offset) && input.dtype() == torch::kFloat32) {
            auto special_tensor = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f, 1.0f, -1.0f
            });
            auto special_result = torch::topk(special_tensor, std::min((int64_t)3, special_tensor.size(0)));
            std::get<0>(special_result).sum();
            std::get<1>(special_result).sum();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
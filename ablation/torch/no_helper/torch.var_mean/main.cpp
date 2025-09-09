#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and data types
        auto input_tensor = generateTensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test 1: Basic var_mean without dim (reduce all dimensions)
        auto result1 = torch::var_mean(input_tensor);
        auto var1 = std::get<0>(result1);
        auto mean1 = std::get<1>(result1);
        
        // Verify results are scalars
        if (var1.dim() != 0 || mean1.dim() != 0) {
            std::cerr << "Expected scalar results for var_mean without dim" << std::endl;
        }

        // Test 2: var_mean with specific dimension
        if (input_tensor.dim() > 0) {
            int dim = consumeIntInRange(Data, Size, offset, 0, input_tensor.dim() - 1);
            
            auto result2 = torch::var_mean(input_tensor, dim);
            auto var2 = std::get<0>(result2);
            auto mean2 = std::get<1>(result2);
            
            // Verify dimension reduction
            auto expected_shape = input_tensor.sizes().vec();
            expected_shape.erase(expected_shape.begin() + dim);
            if (var2.sizes().vec() != expected_shape || mean2.sizes().vec() != expected_shape) {
                std::cerr << "Dimension reduction failed for dim=" << dim << std::endl;
            }
        }

        // Test 3: var_mean with keepdim=true
        if (input_tensor.dim() > 0) {
            int dim = consumeIntInRange(Data, Size, offset, 0, input_tensor.dim() - 1);
            
            auto result3 = torch::var_mean(input_tensor, dim, /*correction=*/1, /*keepdim=*/true);
            auto var3 = std::get<0>(result3);
            auto mean3 = std::get<1>(result3);
            
            // Verify keepdim behavior
            auto expected_shape = input_tensor.sizes().vec();
            expected_shape[dim] = 1;
            if (var3.sizes().vec() != expected_shape || mean3.sizes().vec() != expected_shape) {
                std::cerr << "keepdim=true failed for dim=" << dim << std::endl;
            }
        }

        // Test 4: var_mean with different correction values
        int correction = consumeIntInRange(Data, Size, offset, 0, 5);
        auto result4 = torch::var_mean(input_tensor, /*dim=*/c10::nullopt, correction);
        auto var4 = std::get<0>(result4);
        auto mean4 = std::get<1>(result4);

        // Test 5: var_mean with multiple dimensions (if tensor has enough dims)
        if (input_tensor.dim() >= 2) {
            std::vector<int64_t> dims;
            int num_dims = consumeIntInRange(Data, Size, offset, 1, std::min(3, (int)input_tensor.dim()));
            
            for (int i = 0; i < num_dims; i++) {
                int dim = consumeIntInRange(Data, Size, offset, 0, input_tensor.dim() - 1);
                // Avoid duplicate dimensions
                if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                    dims.push_back(dim);
                }
            }
            
            if (!dims.empty()) {
                auto result5 = torch::var_mean(input_tensor, dims);
                auto var5 = std::get<0>(result5);
                auto mean5 = std::get<1>(result5);
            }
        }

        // Test 6: Edge case - single element tensor
        auto single_elem = torch::randn({1});
        auto result6 = torch::var_mean(single_elem);
        auto var6 = std::get<0>(result6);
        auto mean6 = std::get<1>(result6);

        // Test 7: Edge case - tensor with zero dimension
        if (input_tensor.dim() > 0) {
            auto zero_size_tensor = torch::empty({0});
            if (zero_size_tensor.numel() == 0) {
                try {
                    auto result7 = torch::var_mean(zero_size_tensor);
                } catch (const std::exception& e) {
                    // Expected to potentially fail on empty tensor
                }
            }
        }

        // Test 8: Different data types
        if (input_tensor.scalar_type() != torch::kFloat32) {
            auto float_tensor = input_tensor.to(torch::kFloat32);
            auto result8 = torch::var_mean(float_tensor);
        }

        // Test 9: Large correction value (edge case)
        if (input_tensor.numel() > 1) {
            int large_correction = input_tensor.numel() + 1;
            auto result9 = torch::var_mean(input_tensor, /*dim=*/c10::nullopt, large_correction);
            auto var9 = std::get<0>(result9);
            // Should handle correction >= sample size gracefully
        }

        // Test 10: Negative dimension indexing
        if (input_tensor.dim() > 0) {
            int neg_dim = -consumeIntInRange(Data, Size, offset, 1, input_tensor.dim());
            auto result10 = torch::var_mean(input_tensor, neg_dim);
            auto var10 = std::get<0>(result10);
            auto mean10 = std::get<1>(result10);
        }

        // Test 11: Complex tensor operations
        if (input_tensor.dim() >= 2 && input_tensor.numel() > 4) {
            // Test with reshaped tensor
            auto reshaped = input_tensor.view({-1});
            auto result11 = torch::var_mean(reshaped);
            
            // Test with transposed tensor
            if (input_tensor.dim() == 2) {
                auto transposed = input_tensor.transpose(0, 1);
                auto result12 = torch::var_mean(transposed, 0);
            }
        }

        // Test 12: Verify numerical properties
        if (input_tensor.numel() > 1) {
            auto result_all = torch::var_mean(input_tensor);
            auto var_all = std::get<0>(result_all);
            auto mean_all = std::get<1>(result_all);
            
            // Variance should be non-negative
            if (var_all.item<float>() < 0) {
                std::cerr << "Variance should be non-negative" << std::endl;
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
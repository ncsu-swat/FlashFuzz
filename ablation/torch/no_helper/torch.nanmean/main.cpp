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
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Introduce NaN values randomly in the tensor
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            auto flat_tensor = input_tensor.flatten();
            for (int64_t i = 0; i < flat_tensor.numel(); i++) {
                if (get_random_bool(Data, Size, offset)) {
                    if (input_tensor.dtype() == torch::kFloat32) {
                        flat_tensor[i] = std::numeric_limits<float>::quiet_NaN();
                    } else {
                        flat_tensor[i] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }

        // Test 1: Basic nanmean without dimensions
        auto result1 = torch::nanmean(input_tensor);

        // Test 2: nanmean with random dimension
        if (input_tensor.dim() > 0) {
            int64_t dim = get_random_int(Data, Size, offset) % input_tensor.dim();
            auto result2 = torch::nanmean(input_tensor, dim);
            
            // Test with keepdim=true
            auto result3 = torch::nanmean(input_tensor, dim, /*keepdim=*/true);
        }

        // Test 3: nanmean with multiple dimensions
        if (input_tensor.dim() > 1) {
            std::vector<int64_t> dims;
            int num_dims = (get_random_int(Data, Size, offset) % input_tensor.dim()) + 1;
            for (int i = 0; i < num_dims; i++) {
                int64_t dim = get_random_int(Data, Size, offset) % input_tensor.dim();
                if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                    dims.push_back(dim);
                }
            }
            if (!dims.empty()) {
                auto result4 = torch::nanmean(input_tensor, dims);
                auto result5 = torch::nanmean(input_tensor, dims, /*keepdim=*/true);
            }
        }

        // Test 4: nanmean with dtype conversion
        if (input_tensor.dtype() != torch::kFloat64) {
            auto result6 = torch::nanmean(input_tensor, c10::nullopt, false, torch::kFloat64);
        }

        // Test 5: nanmean with output tensor
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            auto out_tensor = torch::empty({}, input_tensor.options());
            torch::nanmean_out(out_tensor, input_tensor);
        }

        // Test 6: Edge cases - all NaN tensor
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            auto all_nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            auto result7 = torch::nanmean(all_nan_tensor);
        }

        // Test 7: Edge cases - no NaN tensor (should behave like regular mean)
        auto no_nan_tensor = torch::randn_like(input_tensor);
        if (no_nan_tensor.dtype() == torch::kFloat32 || no_nan_tensor.dtype() == torch::kFloat64) {
            auto nanmean_result = torch::nanmean(no_nan_tensor);
            auto mean_result = torch::mean(no_nan_tensor);
        }

        // Test 8: Test with negative dimensions
        if (input_tensor.dim() > 0) {
            int64_t neg_dim = -(get_random_int(Data, Size, offset) % input_tensor.dim() + 1);
            auto result8 = torch::nanmean(input_tensor, neg_dim);
        }

        // Test 9: Test with very large and very small values mixed with NaN
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            auto extreme_tensor = input_tensor.clone();
            auto flat_extreme = extreme_tensor.flatten();
            for (int64_t i = 0; i < flat_extreme.numel(); i++) {
                uint8_t choice = get_random_int(Data, Size, offset) % 4;
                if (choice == 0) {
                    flat_extreme[i] = std::numeric_limits<float>::max();
                } else if (choice == 1) {
                    flat_extreme[i] = std::numeric_limits<float>::lowest();
                } else if (choice == 2) {
                    flat_extreme[i] = std::numeric_limits<float>::quiet_NaN();
                } else if (choice == 3) {
                    flat_extreme[i] = std::numeric_limits<float>::infinity();
                }
            }
            auto result9 = torch::nanmean(extreme_tensor);
        }

        // Test 10: Test with different tensor layouts (contiguous vs non-contiguous)
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            auto result10 = torch::nanmean(transposed);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);
        auto shape = extract_shape(Data, Size, offset);
        
        // Create input tensor with values in valid range for erfinv
        // erfinv is only defined for values in (-1, 1)
        torch::Tensor input;
        
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // Create tensor with random values
            input = create_tensor(Data, Size, offset, shape, dtype, device);
            
            // Clamp values to valid range (-1, 1) for erfinv
            // Use slightly smaller range to avoid edge cases at exactly -1 and 1
            input = torch::clamp(input, -0.99, 0.99);
        } else {
            // For non-floating point types, create and convert to float
            auto temp_tensor = create_tensor(Data, Size, offset, shape, torch::kFloat32, device);
            input = torch::clamp(temp_tensor, -0.99, 0.99);
            if (dtype != torch::kFloat32) {
                input = input.to(dtype);
            }
        }

        // Test basic erfinv operation
        auto result1 = torch::erfinv(input);

        // Test with output tensor
        auto out_tensor = torch::empty_like(input);
        auto result2 = torch::erfinv(input, out_tensor);

        // Test edge cases with specific values if we have enough data
        if (offset + 4 < Size) {
            uint32_t edge_case = extract_uint32(Data, Size, offset);
            
            switch (edge_case % 6) {
                case 0: {
                    // Test with zeros
                    auto zero_input = torch::zeros_like(input);
                    auto zero_result = torch::erfinv(zero_input);
                    break;
                }
                case 1: {
                    // Test with small positive values
                    auto small_pos = torch::full_like(input, 0.1);
                    auto small_pos_result = torch::erfinv(small_pos);
                    break;
                }
                case 2: {
                    // Test with small negative values
                    auto small_neg = torch::full_like(input, -0.1);
                    auto small_neg_result = torch::erfinv(small_neg);
                    break;
                }
                case 3: {
                    // Test with values close to 1
                    auto near_one = torch::full_like(input, 0.9);
                    auto near_one_result = torch::erfinv(near_one);
                    break;
                }
                case 4: {
                    // Test with values close to -1
                    auto near_neg_one = torch::full_like(input, -0.9);
                    auto near_neg_one_result = torch::erfinv(near_neg_one);
                    break;
                }
                case 5: {
                    // Test with mixed values
                    auto mixed = torch::randn_like(input) * 0.5; // Scale to keep in valid range
                    auto mixed_result = torch::erfinv(mixed);
                    break;
                }
            }
        }

        // Test different tensor shapes if we have more data
        if (offset + 4 < Size) {
            uint32_t shape_test = extract_uint32(Data, Size, offset);
            
            switch (shape_test % 4) {
                case 0: {
                    // Test scalar
                    auto scalar_input = torch::tensor(0.5, torch::dtype(dtype).device(device));
                    auto scalar_result = torch::erfinv(scalar_input);
                    break;
                }
                case 1: {
                    // Test 1D tensor
                    auto vec_input = torch::linspace(-0.8, 0.8, 10, torch::dtype(dtype).device(device));
                    auto vec_result = torch::erfinv(vec_input);
                    break;
                }
                case 2: {
                    // Test 2D tensor
                    auto mat_input = torch::randn({3, 4}, torch::dtype(dtype).device(device)) * 0.7;
                    auto mat_result = torch::erfinv(mat_input);
                    break;
                }
                case 3: {
                    // Test higher dimensional tensor
                    auto high_dim_input = torch::randn({2, 3, 2}, torch::dtype(dtype).device(device)) * 0.6;
                    auto high_dim_result = torch::erfinv(high_dim_input);
                    break;
                }
            }
        }

        // Test gradient computation if input requires grad
        if (input.dtype().is_floating_point() && offset + 1 < Size) {
            bool requires_grad = (Data[offset] % 2) == 1;
            offset++;
            
            if (requires_grad) {
                auto grad_input = input.clone().detach().requires_grad_(true);
                auto grad_output = torch::erfinv(grad_input);
                
                // Compute gradients
                auto grad_outputs = torch::ones_like(grad_output);
                grad_output.backward(grad_outputs);
            }
        }

        // Verify output properties
        if (result1.defined()) {
            // Check that output has same shape as input
            if (!result1.sizes().equals(input.sizes())) {
                throw std::runtime_error("Output shape mismatch");
            }
            
            // Check that output dtype matches input dtype for floating point
            if (input.dtype().is_floating_point() && result1.dtype() != input.dtype()) {
                throw std::runtime_error("Output dtype mismatch");
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
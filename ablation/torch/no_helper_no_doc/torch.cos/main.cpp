#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor parameters
        auto tensor_params = generate_tensor_params(Data, Size, offset);
        if (!tensor_params.has_value()) {
            return 0;
        }

        auto [shape, dtype, device_type, requires_grad] = tensor_params.value();

        // Create input tensor with various edge case values
        torch::Tensor input;
        
        // Try different tensor creation strategies based on remaining data
        if (offset < Size) {
            uint8_t creation_strategy = Data[offset++];
            
            switch (creation_strategy % 6) {
                case 0: {
                    // Random tensor
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    break;
                }
                case 1: {
                    // Tensor with extreme values
                    input = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        input.fill_(std::numeric_limits<float>::max());
                    } else if (dtype == torch::kInt32) {
                        input.fill_(std::numeric_limits<int32_t>::max());
                    } else if (dtype == torch::kInt64) {
                        input.fill_(std::numeric_limits<int64_t>::max());
                    }
                    break;
                }
                case 2: {
                    // Tensor with special float values (inf, -inf, nan)
                    input = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        if (offset < Size) {
                            uint8_t special_val = Data[offset++] % 3;
                            switch (special_val) {
                                case 0: input.fill_(std::numeric_limits<float>::infinity()); break;
                                case 1: input.fill_(-std::numeric_limits<float>::infinity()); break;
                                case 2: input.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                            }
                        }
                    } else {
                        input.fill_(0);
                    }
                    break;
                }
                case 3: {
                    // Zero tensor
                    input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    break;
                }
                case 4: {
                    // Ones tensor
                    input = torch::ones(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    break;
                }
                case 5: {
                    // Tensor with values from fuzzer data
                    input = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
                    auto flat_input = input.flatten();
                    int64_t num_elements = flat_input.numel();
                    
                    for (int64_t i = 0; i < num_elements && offset < Size; ++i) {
                        if (dtype == torch::kFloat32) {
                            float val = extract_float(Data, Size, offset);
                            flat_input[i] = val;
                        } else if (dtype == torch::kFloat64) {
                            double val = extract_double(Data, Size, offset);
                            flat_input[i] = val;
                        } else if (dtype == torch::kInt32) {
                            int32_t val = extract_int32(Data, Size, offset);
                            flat_input[i] = val;
                        } else if (dtype == torch::kInt64) {
                            int64_t val = extract_int64(Data, Size, offset);
                            flat_input[i] = val;
                        }
                    }
                    break;
                }
            }
        } else {
            // Default case when no more data
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device_type).requires_grad(requires_grad));
        }

        // Test torch::cos with the generated input
        torch::Tensor result = torch::cos(input);

        // Verify result properties
        if (result.sizes() != input.sizes()) {
            std::cerr << "Output shape mismatch" << std::endl;
        }

        // Test in-place version if applicable
        if (!requires_grad && (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
            torch::Tensor input_copy = input.clone();
            input_copy.cos_();
            
            // Verify in-place operation worked correctly
            if (!torch::allclose(result, input_copy, 1e-5, 1e-8, /*equal_nan=*/true)) {
                std::cerr << "In-place cos operation mismatch" << std::endl;
            }
        }

        // Test gradient computation if requires_grad is true
        if (requires_grad && input.requires_grad()) {
            torch::Tensor loss = result.sum();
            loss.backward();
            
            // Check that gradients were computed
            if (!input.grad().defined()) {
                std::cerr << "Gradients not computed" << std::endl;
            }
        }

        // Test with different tensor layouts/strides if possible
        if (input.dim() >= 2) {
            torch::Tensor transposed = input.transpose(0, 1);
            torch::Tensor transposed_result = torch::cos(transposed);
            
            // Verify the operation works with non-contiguous tensors
            if (transposed_result.sizes() != transposed.sizes()) {
                std::cerr << "Transposed tensor cos failed" << std::endl;
            }
        }

        // Test edge cases for specific input ranges
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // Test with very large values that might cause numerical issues
            torch::Tensor large_input = torch::tensor({1e10, -1e10, 1e20, -1e20}, 
                torch::TensorOptions().dtype(dtype).device(device_type));
            torch::Tensor large_result = torch::cos(large_input);
            
            // Test with values near multiples of pi
            torch::Tensor pi_input = torch::tensor({M_PI, 2*M_PI, M_PI/2, 3*M_PI/2}, 
                torch::TensorOptions().dtype(dtype).device(device_type));
            torch::Tensor pi_result = torch::cos(pi_input);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
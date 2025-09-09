#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensor creation and value
        if (Size < 32) return 0;

        // Extract tensor shapes and properties
        auto input_shape = extract_tensor_shape(Data, Size, offset, 4);
        auto tensor1_shape = extract_tensor_shape(Data, Size, offset, 4);
        auto tensor2_shape = extract_tensor_shape(Data, Size, offset, 4);
        
        if (offset >= Size) return 0;

        // Extract dtype (limit to float types since addcdiv works best with them)
        uint8_t dtype_idx = Data[offset++] % 3;
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }

        if (offset >= Size) return 0;

        // Extract value parameter
        double value = extract_float_value(Data, Size, offset);
        
        // Create tensors with different initialization strategies
        uint8_t init_strategy = Data[offset++] % 4;
        
        torch::Tensor input, tensor1, tensor2;
        
        switch (init_strategy) {
            case 0:
                // Random tensors
                input = torch::randn(input_shape, torch::dtype(dtype));
                tensor1 = torch::randn(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::randn(tensor2_shape, torch::dtype(dtype));
                break;
            case 1:
                // Small positive values to avoid extreme results
                input = torch::rand(input_shape, torch::dtype(dtype)) * 2 - 1;
                tensor1 = torch::rand(tensor1_shape, torch::dtype(dtype)) * 2 - 1;
                tensor2 = torch::rand(tensor2_shape, torch::dtype(dtype)) * 0.5 + 0.5; // Avoid zeros
                break;
            case 2:
                // Include some edge values
                input = create_edge_case_tensor(input_shape, dtype, Data, offset);
                tensor1 = create_edge_case_tensor(tensor1_shape, dtype, Data, offset);
                tensor2 = create_edge_case_tensor(tensor2_shape, dtype, Data, offset);
                // Ensure tensor2 doesn't have zeros to avoid division by zero
                tensor2 = torch::where(torch::abs(tensor2) < 1e-6, 
                                     torch::ones_like(tensor2), tensor2);
                break;
            case 3:
                // Mixed: some zeros, ones, and random values
                input = torch::zeros(input_shape, torch::dtype(dtype));
                tensor1 = torch::ones(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::full(tensor2_shape, 2.0, torch::dtype(dtype));
                break;
        }

        // Ensure tensor2 doesn't contain zeros (avoid division by zero)
        tensor2 = torch::where(torch::abs(tensor2) < 1e-7, 
                              torch::sign(tensor2) * 1e-7, tensor2);

        if (offset < Size) {
            uint8_t test_variant = Data[offset++] % 8;
            
            switch (test_variant) {
                case 0:
                    // Basic addcdiv
                    {
                        auto result = torch::addcdiv(input, tensor1, tensor2);
                        validate_tensor_result(result);
                    }
                    break;
                case 1:
                    // With custom value
                    {
                        auto result = torch::addcdiv(input, tensor1, tensor2, value);
                        validate_tensor_result(result);
                    }
                    break;
                case 2:
                    // With output tensor
                    {
                        auto out_shape = torch::broadcast_tensors({input, tensor1, tensor2})[0].sizes();
                        auto out = torch::empty(out_shape, torch::dtype(dtype));
                        torch::addcdiv_out(out, input, tensor1, tensor2, value);
                        validate_tensor_result(out);
                    }
                    break;
                case 3:
                    // In-place operation (addcdiv_)
                    {
                        auto input_copy = input.clone();
                        input_copy.addcdiv_(tensor1, tensor2, value);
                        validate_tensor_result(input_copy);
                    }
                    break;
                case 4:
                    // Test with broadcasting - different shapes
                    {
                        auto scalar_tensor1 = torch::tensor(extract_float_value(Data, Size, offset), dtype);
                        auto result = torch::addcdiv(input, scalar_tensor1, tensor2, value);
                        validate_tensor_result(result);
                    }
                    break;
                case 5:
                    // Test with very small values
                    {
                        auto small_value = 1e-8;
                        auto result = torch::addcdiv(input, tensor1, tensor2, small_value);
                        validate_tensor_result(result);
                    }
                    break;
                case 6:
                    // Test with large values
                    {
                        auto large_value = 1e6;
                        auto result = torch::addcdiv(input, tensor1, tensor2, large_value);
                        validate_tensor_result(result);
                    }
                    break;
                case 7:
                    // Test with negative value
                    {
                        auto neg_value = -std::abs(value);
                        auto result = torch::addcdiv(input, tensor1, tensor2, neg_value);
                        validate_tensor_result(result);
                    }
                    break;
            }
        }

        // Additional edge case testing
        if (offset < Size) {
            uint8_t edge_test = Data[offset++] % 4;
            
            switch (edge_test) {
                case 0:
                    // Test with zero input
                    {
                        auto zero_input = torch::zeros_like(input);
                        auto result = torch::addcdiv(zero_input, tensor1, tensor2, value);
                        validate_tensor_result(result);
                    }
                    break;
                case 1:
                    // Test with ones
                    {
                        auto ones_tensor1 = torch::ones_like(tensor1);
                        auto ones_tensor2 = torch::ones_like(tensor2);
                        auto result = torch::addcdiv(input, ones_tensor1, ones_tensor2, value);
                        validate_tensor_result(result);
                    }
                    break;
                case 2:
                    // Test with same tensor for tensor1 and tensor2 (should give value + input)
                    {
                        auto same_tensor = torch::rand_like(tensor1) + 0.1; // Avoid zeros
                        auto result = torch::addcdiv(input, same_tensor, same_tensor, value);
                        validate_tensor_result(result);
                    }
                    break;
                case 3:
                    // Test gradient computation if tensors require grad
                    {
                        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                            auto grad_input = input.clone().requires_grad_(true);
                            auto grad_tensor1 = tensor1.clone().requires_grad_(true);
                            auto grad_tensor2 = tensor2.clone().requires_grad_(true);
                            
                            auto result = torch::addcdiv(grad_input, grad_tensor1, grad_tensor2, value);
                            auto loss = result.sum();
                            loss.backward();
                            
                            validate_tensor_result(grad_input.grad());
                            validate_tensor_result(grad_tensor1.grad());
                            validate_tensor_result(grad_tensor2.grad());
                        }
                    }
                    break;
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
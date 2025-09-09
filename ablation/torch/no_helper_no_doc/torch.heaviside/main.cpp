#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to work with
        if (Size < 16) {
            return 0;
        }

        // Generate input tensor dimensions and properties
        auto input_dims = generateRandomDims(Data, Size, offset, 1, 4);
        auto input_dtype = generateRandomDtype(Data, Size, offset);
        auto input_device = generateRandomDevice(Data, Size, offset);
        
        // Generate values tensor dimensions and properties
        auto values_dims = generateRandomDims(Data, Size, offset, 1, 4);
        auto values_dtype = generateRandomDtype(Data, Size, offset);
        auto values_device = generateRandomDevice(Data, Size, offset);

        // Create input tensor
        torch::Tensor input = generateRandomTensor(input_dims, input_dtype, input_device, Data, Size, offset);
        
        // Create values tensor
        torch::Tensor values = generateRandomTensor(values_dims, values_dtype, values_device, Data, Size, offset);

        // Test basic heaviside operation
        torch::Tensor result1 = torch::heaviside(input, values);

        // Test with broadcasting - make values a scalar sometimes
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor scalar_values = torch::scalar_tensor(generateRandomFloat(Data, Size, offset), 
                                                              torch::TensorOptions().dtype(values_dtype).device(values_device));
            torch::Tensor result2 = torch::heaviside(input, scalar_values);
        }

        // Test with different broadcasting scenarios
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Create values tensor with different but broadcastable shape
            std::vector<int64_t> broadcast_dims;
            for (size_t i = 0; i < input_dims.size(); ++i) {
                if (offset < Size && Data[offset++] % 2 == 0) {
                    broadcast_dims.push_back(1);
                } else {
                    broadcast_dims.push_back(input_dims[i]);
                }
            }
            torch::Tensor broadcast_values = generateRandomTensor(broadcast_dims, values_dtype, values_device, Data, Size, offset);
            torch::Tensor result3 = torch::heaviside(input, broadcast_values);
        }

        // Test with special values (zeros, ones, negative values)
        if (offset < Size && Data[offset++] % 4 == 0) {
            torch::Tensor zeros_input = torch::zeros_like(input);
            torch::Tensor result4 = torch::heaviside(zeros_input, values);
        }

        if (offset < Size && Data[offset++] % 4 == 1) {
            torch::Tensor ones_input = torch::ones_like(input);
            torch::Tensor result5 = torch::heaviside(ones_input, values);
        }

        if (offset < Size && Data[offset++] % 4 == 2) {
            torch::Tensor neg_input = -torch::abs(input);
            torch::Tensor result6 = torch::heaviside(neg_input, values);
        }

        // Test with different data types if possible
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                auto new_dtype = generateRandomDtype(Data, Size, offset);
                torch::Tensor converted_input = input.to(new_dtype);
                torch::Tensor converted_values = values.to(new_dtype);
                torch::Tensor result7 = torch::heaviside(converted_input, converted_values);
            } catch (...) {
                // Some dtype conversions might not be supported, ignore
            }
        }

        // Test with very small and very large values
        if (offset < Size && Data[offset++] % 5 == 0) {
            torch::Tensor small_input = input * 1e-10;
            torch::Tensor result8 = torch::heaviside(small_input, values);
        }

        if (offset < Size && Data[offset++] % 5 == 1) {
            torch::Tensor large_input = input * 1e10;
            torch::Tensor result9 = torch::heaviside(large_input, values);
        }

        // Test with inf and -inf values
        if (offset < Size && Data[offset++] % 6 == 0) {
            torch::Tensor inf_input = torch::full_like(input, std::numeric_limits<float>::infinity());
            torch::Tensor result10 = torch::heaviside(inf_input, values);
        }

        if (offset < Size && Data[offset++] % 6 == 1) {
            torch::Tensor neg_inf_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
            torch::Tensor result11 = torch::heaviside(neg_inf_input, values);
        }

        // Test with NaN values
        if (offset < Size && Data[offset++] % 7 == 0) {
            torch::Tensor nan_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor result12 = torch::heaviside(nan_input, values);
        }

        if (offset < Size && Data[offset++] % 7 == 1) {
            torch::Tensor nan_values = torch::full_like(values, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor result13 = torch::heaviside(input, nan_values);
        }

        // Test in-place operation if supported
        if (offset < Size && Data[offset++] % 8 == 0) {
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.heaviside_(values);
            } catch (...) {
                // In-place operation might not be supported for all cases
            }
        }

        // Test with requires_grad
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor grad_input = input.clone().requires_grad_(true);
                torch::Tensor grad_values = values.clone().requires_grad_(true);
                torch::Tensor result14 = torch::heaviside(grad_input, grad_values);
                
                // Try backward pass
                if (result14.numel() > 0) {
                    torch::Tensor grad_output = torch::ones_like(result14);
                    result14.backward(grad_output);
                }
            } catch (...) {
                // Gradient computation might not be supported
            }
        }

        // Test with empty tensors
        if (offset < Size && Data[offset++] % 10 == 0) {
            torch::Tensor empty_input = torch::empty({0}, input.options());
            torch::Tensor empty_values = torch::empty({0}, values.options());
            torch::Tensor result15 = torch::heaviside(empty_input, empty_values);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
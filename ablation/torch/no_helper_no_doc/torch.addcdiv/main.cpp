#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and properties
        auto input_dims = parse_tensor_dims(Data, Size, offset);
        if (input_dims.empty()) return 0;

        auto tensor1_dims = parse_tensor_dims(Data, Size, offset);
        if (tensor1_dims.empty()) return 0;

        auto tensor2_dims = parse_tensor_dims(Data, Size, offset);
        if (tensor2_dims.empty()) return 0;

        // Parse scalar value for multiplication
        if (offset + sizeof(float) > Size) return 0;
        float value = *reinterpret_cast<const float*>(Data + offset);
        offset += sizeof(float);

        // Parse dtype
        auto dtype = parse_dtype(Data, Size, offset);

        // Parse device type
        auto device = parse_device(Data, Size, offset);

        // Create input tensor
        auto input = create_tensor(input_dims, dtype, device);
        if (!input.defined()) return 0;

        // Create tensor1 (numerator)
        auto tensor1 = create_tensor(tensor1_dims, dtype, device);
        if (!tensor1.defined()) return 0;

        // Create tensor2 (denominator) - avoid zeros to prevent division by zero
        auto tensor2 = create_tensor(tensor2_dims, dtype, device);
        if (!tensor2.defined()) return 0;
        
        // Add small epsilon to avoid division by zero
        tensor2 = tensor2 + 1e-6;

        // Test basic addcdiv operation: input + value * (tensor1 / tensor2)
        auto result1 = torch::addcdiv(input, tensor1, tensor2, value);

        // Test with default value (1.0)
        auto result2 = torch::addcdiv(input, tensor1, tensor2);

        // Test in-place version if input is contiguous and writable
        if (input.is_contiguous() && !input.requires_grad()) {
            auto input_copy = input.clone();
            input_copy.addcdiv_(tensor1, tensor2, value);
        }

        // Test with broadcasting - create smaller tensors that can broadcast
        if (input_dims.size() > 1) {
            std::vector<int64_t> broadcast_dims = {input_dims.back()};
            auto broadcast_tensor1 = create_tensor(broadcast_dims, dtype, device);
            auto broadcast_tensor2 = create_tensor(broadcast_dims, dtype, device);
            
            if (broadcast_tensor1.defined() && broadcast_tensor2.defined()) {
                broadcast_tensor2 = broadcast_tensor2 + 1e-6; // Avoid division by zero
                auto broadcast_result = torch::addcdiv(input, broadcast_tensor1, broadcast_tensor2, value);
            }
        }

        // Test with scalar tensors
        auto scalar_tensor1 = torch::tensor(2.0, torch::TensorOptions().dtype(dtype).device(device));
        auto scalar_tensor2 = torch::tensor(3.0, torch::TensorOptions().dtype(dtype).device(device));
        auto scalar_result = torch::addcdiv(input, scalar_tensor1, scalar_tensor2, value);

        // Test edge cases with different values
        std::vector<float> test_values = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f};
        for (float test_val : test_values) {
            if (offset >= Size) break;
            auto edge_result = torch::addcdiv(input, tensor1, tensor2, test_val);
        }

        // Test with different tensor combinations to explore broadcasting
        try {
            // Test when input is scalar
            auto scalar_input = torch::tensor(1.0, torch::TensorOptions().dtype(dtype).device(device));
            auto scalar_input_result = torch::addcdiv(scalar_input, tensor1, tensor2, value);
        } catch (...) {
            // Broadcasting might fail, continue testing
        }

        // Test output parameter version if available
        auto output_tensor = torch::empty_like(input);
        torch::addcdiv_out(output_tensor, input, tensor1, tensor2, value);

        // Verify results are finite (not NaN or Inf) when possible
        if (result1.dtype().isFloatingType()) {
            auto finite_check = torch::isfinite(result1);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
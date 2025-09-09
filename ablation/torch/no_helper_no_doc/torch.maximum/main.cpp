#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract parameters for tensor creation
        auto shape1 = extract_tensor_shape(Data, Size, offset);
        auto shape2 = extract_tensor_shape(Data, Size, offset);
        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);

        // Create first tensor
        torch::Tensor tensor1;
        if (shape1.empty()) {
            // Scalar tensor
            tensor1 = create_random_tensor({}, dtype, device, Data, Size, offset);
        } else {
            tensor1 = create_random_tensor(shape1, dtype, device, Data, Size, offset);
        }

        // Create second tensor - test broadcasting scenarios
        torch::Tensor tensor2;
        if (offset < Size) {
            uint8_t broadcast_choice = Data[offset++] % 4;
            
            switch (broadcast_choice) {
                case 0:
                    // Same shape
                    tensor2 = create_random_tensor(tensor1.sizes().vec(), dtype, device, Data, Size, offset);
                    break;
                case 1:
                    // Scalar
                    tensor2 = create_random_tensor({}, dtype, device, Data, Size, offset);
                    break;
                case 2:
                    // Different but broadcastable shape
                    if (!shape2.empty()) {
                        tensor2 = create_random_tensor(shape2, dtype, device, Data, Size, offset);
                    } else {
                        tensor2 = create_random_tensor({1}, dtype, device, Data, Size, offset);
                    }
                    break;
                case 3:
                    // Single element tensor
                    tensor2 = create_random_tensor({1}, dtype, device, Data, Size, offset);
                    break;
            }
        } else {
            tensor2 = create_random_tensor(tensor1.sizes().vec(), dtype, device, Data, Size, offset);
        }

        // Test torch::maximum with two tensors
        auto result1 = torch::maximum(tensor1, tensor2);
        
        // Verify result properties
        if (result1.device() != tensor1.device()) {
            throw std::runtime_error("Result device mismatch");
        }

        // Test with scalar values if we have remaining data
        if (offset < Size) {
            double scalar_val = extract_float_value(Data, Size, offset);
            
            // Test tensor-scalar maximum
            auto result2 = torch::maximum(tensor1, scalar_val);
            auto result3 = torch::maximum(scalar_val, tensor1);
            
            // Verify scalar results
            if (result2.device() != tensor1.device() || result3.device() != tensor1.device()) {
                throw std::runtime_error("Scalar result device mismatch");
            }
        }

        // Test edge cases with special values if dtype supports them
        if (tensor1.dtype() == torch::kFloat32 || tensor1.dtype() == torch::kFloat64) {
            // Test with infinity
            auto inf_tensor = torch::full_like(tensor1, std::numeric_limits<double>::infinity());
            auto result_inf = torch::maximum(tensor1, inf_tensor);
            
            // Test with negative infinity
            auto neg_inf_tensor = torch::full_like(tensor1, -std::numeric_limits<double>::infinity());
            auto result_neg_inf = torch::maximum(tensor1, neg_inf_tensor);
            
            // Test with NaN
            auto nan_tensor = torch::full_like(tensor1, std::numeric_limits<double>::quiet_NaN());
            auto result_nan = torch::maximum(tensor1, nan_tensor);
        }

        // Test with zero tensors
        auto zero_tensor = torch::zeros_like(tensor1);
        auto result_zero = torch::maximum(tensor1, zero_tensor);

        // Test with negative values
        auto neg_tensor = -torch::abs(tensor1);
        auto result_neg = torch::maximum(tensor1, neg_tensor);

        // Test in-place operations if tensors are compatible
        if (tensor1.sizes() == tensor2.sizes() && tensor1.dtype() == tensor2.dtype()) {
            auto tensor1_copy = tensor1.clone();
            tensor1_copy.maximum_(tensor2);
        }

        // Test with different tensor orders (C vs Fortran contiguous)
        if (tensor1.dim() > 1) {
            auto transposed = tensor1.transpose(0, -1);
            auto result_transposed = torch::maximum(transposed, tensor2);
        }

        // Test with sliced tensors
        if (tensor1.numel() > 1) {
            auto sliced = tensor1.flatten().slice(0, 0, std::min(tensor1.numel(), 10L));
            auto sliced2 = tensor2.flatten().slice(0, 0, std::min(tensor2.numel(), 10L));
            if (sliced.sizes() == sliced2.sizes()) {
                auto result_sliced = torch::maximum(sliced, sliced2);
            }
        }

        // Force evaluation of lazy tensors
        result1.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
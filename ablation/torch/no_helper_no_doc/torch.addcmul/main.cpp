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
        float value = parse_float(Data, Size, offset);

        // Parse dtype
        auto dtype = parse_dtype(Data, Size, offset);

        // Parse device type
        auto device = parse_device(Data, Size, offset);

        // Create input tensor
        auto input = create_tensor(input_dims, dtype, device);
        if (!input.defined()) return 0;

        // Create tensor1 for multiplication
        auto tensor1 = create_tensor(tensor1_dims, dtype, device);
        if (!tensor1.defined()) return 0;

        // Create tensor2 for multiplication
        auto tensor2 = create_tensor(tensor2_dims, dtype, device);
        if (!tensor2.defined()) return 0;

        // Test basic addcmul operation: input + value * tensor1 * tensor2
        auto result1 = torch::addcmul(input, tensor1, tensor2, value);

        // Test addcmul with default value (1.0)
        auto result2 = torch::addcmul(input, tensor1, tensor2);

        // Test in-place addcmul
        auto input_copy = input.clone();
        input_copy.addcmul_(tensor1, tensor2, value);

        // Test with different scalar values
        if (offset < Size) {
            float value2 = parse_float(Data, Size, offset);
            auto result3 = torch::addcmul(input, tensor1, tensor2, value2);
        }

        // Test with broadcasting - create tensors with different but compatible shapes
        if (offset < Size) {
            auto broadcast_dims1 = parse_tensor_dims(Data, Size, offset);
            auto broadcast_dims2 = parse_tensor_dims(Data, Size, offset);
            
            if (!broadcast_dims1.empty() && !broadcast_dims2.empty()) {
                auto broadcast_tensor1 = create_tensor(broadcast_dims1, dtype, device);
                auto broadcast_tensor2 = create_tensor(broadcast_dims2, dtype, device);
                
                if (broadcast_tensor1.defined() && broadcast_tensor2.defined()) {
                    try {
                        auto broadcast_result = torch::addcmul(input, broadcast_tensor1, broadcast_tensor2, value);
                    } catch (const std::exception&) {
                        // Broadcasting might fail, which is expected for incompatible shapes
                    }
                }
            }
        }

        // Test with zero-dimensional tensors
        if (offset < Size) {
            auto scalar_tensor1 = torch::tensor(parse_float(Data, Size, offset), torch::TensorOptions().dtype(dtype).device(device));
            auto scalar_tensor2 = torch::tensor(parse_float(Data, Size, offset), torch::TensorOptions().dtype(dtype).device(device));
            
            try {
                auto scalar_result = torch::addcmul(input, scalar_tensor1, scalar_tensor2, value);
            } catch (const std::exception&) {
                // May fail due to broadcasting rules
            }
        }

        // Test with negative values
        auto result_neg = torch::addcmul(input, tensor1, tensor2, -value);

        // Test with very small and very large values
        if (offset < Size) {
            float small_val = parse_float(Data, Size, offset) * 1e-10f;
            float large_val = parse_float(Data, Size, offset) * 1e10f;
            
            auto result_small = torch::addcmul(input, tensor1, tensor2, small_val);
            auto result_large = torch::addcmul(input, tensor1, tensor2, large_val);
        }

        // Test with different tensor orders/strides
        if (input.dim() >= 2) {
            auto transposed_input = input.transpose(0, 1);
            auto result_transposed = torch::addcmul(transposed_input, tensor1, tensor2, value);
        }

        // Test with contiguous and non-contiguous tensors
        if (tensor1.dim() >= 2) {
            auto non_contiguous1 = tensor1.transpose(0, 1);
            auto non_contiguous2 = tensor2.dim() >= 2 ? tensor2.transpose(0, 1) : tensor2;
            
            try {
                auto result_non_contiguous = torch::addcmul(input, non_contiguous1, non_contiguous2, value);
            } catch (const std::exception&) {
                // May fail due to shape incompatibility
            }
        }

        // Test edge case: zero value
        auto result_zero = torch::addcmul(input, tensor1, tensor2, 0.0f);

        // Test with special float values if applicable
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            auto result_inf = torch::addcmul(input, tensor1, tensor2, std::numeric_limits<float>::infinity());
            auto result_ninf = torch::addcmul(input, tensor1, tensor2, -std::numeric_limits<float>::infinity());
            
            // Test with NaN (might produce NaN results)
            try {
                auto result_nan = torch::addcmul(input, tensor1, tensor2, std::numeric_limits<float>::quiet_NaN());
            } catch (const std::exception&) {
                // NaN operations might throw in some cases
            }
        }

        // Verify results are defined and have expected properties
        if (result1.defined()) {
            auto sizes = result1.sizes();
            auto result_dtype = result1.dtype();
            auto result_device = result1.device();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
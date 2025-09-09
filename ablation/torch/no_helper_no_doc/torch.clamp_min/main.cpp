#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and min value
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (!tensor_info.has_value()) {
            return 0;
        }

        // Create input tensor with various data types and shapes
        torch::Tensor input_tensor;
        try {
            input_tensor = create_tensor_from_data(Data, Size, offset, tensor_info.value());
        } catch (...) {
            return 0;
        }

        // Extract min value - try different approaches
        if (offset >= Size) {
            return 0;
        }

        // Test with scalar min value
        double min_scalar = extract_float_value(Data, Size, offset);
        
        // Test torch::clamp_min with scalar
        auto result1 = torch::clamp_min(input_tensor, min_scalar);
        
        // Test in-place version if tensor allows it
        if (input_tensor.is_contiguous() && !input_tensor.is_view()) {
            try {
                auto input_copy = input_tensor.clone();
                torch::clamp_min_(input_copy, min_scalar);
            } catch (...) {
                // In-place operation might fail for some tensor types
            }
        }

        // Test with tensor min value if we have enough data
        if (offset < Size - 8) {
            try {
                // Create a min tensor with compatible shape
                auto min_tensor_shape = input_tensor.sizes().vec();
                
                // Try broadcasting - create scalar tensor
                auto min_tensor = torch::full({}, min_scalar, input_tensor.options());
                auto result2 = torch::clamp_min(input_tensor, min_tensor);
                
                // Try with same shape tensor if reasonable size
                if (input_tensor.numel() <= 1000) {
                    auto min_tensor_same_shape = torch::full_like(input_tensor, min_scalar);
                    auto result3 = torch::clamp_min(input_tensor, min_tensor_same_shape);
                }
                
                // Try with broadcastable shapes
                if (min_tensor_shape.size() > 0) {
                    // Create tensor with last dimension = 1 for broadcasting
                    auto broadcast_shape = min_tensor_shape;
                    broadcast_shape.back() = 1;
                    if (torch::numel(broadcast_shape) <= 100) {
                        auto min_broadcast = torch::full(broadcast_shape, min_scalar, input_tensor.options());
                        auto result4 = torch::clamp_min(input_tensor, min_broadcast);
                    }
                }
            } catch (...) {
                // Broadcasting or tensor creation might fail
            }
        }

        // Test edge cases with special values
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            try {
                // Test with infinity
                auto result_inf = torch::clamp_min(input_tensor, std::numeric_limits<double>::infinity());
                auto result_neg_inf = torch::clamp_min(input_tensor, -std::numeric_limits<double>::infinity());
                
                // Test with NaN (should handle gracefully)
                auto result_nan = torch::clamp_min(input_tensor, std::numeric_limits<double>::quiet_NaN());
            } catch (...) {
                // Special values might cause issues
            }
        }

        // Test with different tensor properties
        if (input_tensor.numel() > 0) {
            try {
                // Test with non-contiguous tensor
                if (input_tensor.dim() > 1) {
                    auto transposed = input_tensor.transpose(0, -1);
                    auto result_non_contig = torch::clamp_min(transposed, min_scalar);
                }
                
                // Test with sliced tensor
                if (input_tensor.size(0) > 1) {
                    auto sliced = input_tensor.slice(0, 0, input_tensor.size(0) / 2);
                    auto result_sliced = torch::clamp_min(sliced, min_scalar);
                }
            } catch (...) {
                // Tensor manipulations might fail
            }
        }

        // Test type promotion scenarios
        if (offset < Size - 4) {
            try {
                // Test with different scalar types
                int min_int = static_cast<int>(min_scalar);
                auto result_int = torch::clamp_min(input_tensor, min_int);
                
                float min_float = static_cast<float>(min_scalar);
                auto result_float = torch::clamp_min(input_tensor, min_float);
            } catch (...) {
                // Type promotion might fail
            }
        }

        // Test with zero-sized tensors
        try {
            auto empty_tensor = torch::empty({0}, input_tensor.options());
            auto result_empty = torch::clamp_min(empty_tensor, min_scalar);
        } catch (...) {
            // Empty tensor operations might have special behavior
        }

        // Test gradient computation if tensor requires grad
        if (input_tensor.dtype().is_floating_point()) {
            try {
                auto input_with_grad = input_tensor.detach().requires_grad_(true);
                auto result_grad = torch::clamp_min(input_with_grad, min_scalar);
                
                if (result_grad.numel() > 0) {
                    auto grad_output = torch::ones_like(result_grad);
                    result_grad.backward(grad_output);
                }
            } catch (...) {
                // Gradient computation might fail
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
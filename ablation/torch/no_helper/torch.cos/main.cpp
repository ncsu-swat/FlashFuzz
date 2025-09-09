#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor shape (1-4 dimensions)
        auto shape = generate_tensor_shape(Data, Size, offset, 1, 4);
        if (shape.empty()) return 0;

        // Generate dtype - focus on floating point types since cos is mathematical
        auto dtype_options = {torch::kFloat32, torch::kFloat64, torch::kFloat16, torch::kBFloat16};
        auto dtype = generate_dtype_from_options(Data, Size, offset, dtype_options);

        // Generate device
        auto device = generate_device(Data, Size, offset);

        // Create input tensor with various value ranges to test edge cases
        torch::Tensor input;
        
        // Choose value generation strategy
        uint8_t value_strategy = consume_uint8_t(Data, Size, offset);
        
        switch (value_strategy % 6) {
            case 0:
                // Normal random values
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            case 1:
                // Large values to test numerical stability
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 100.0;
                break;
            case 2:
                // Small values near zero
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 0.01;
                break;
            case 3:
                // Values around pi multiples (important for cosine)
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * M_PI;
                break;
            case 4:
                // Special values including inf, -inf
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                if (input.numel() > 0) {
                    auto flat = input.flatten();
                    if (flat.size(0) > 0) flat[0] = std::numeric_limits<float>::infinity();
                    if (flat.size(0) > 1) flat[1] = -std::numeric_limits<float>::infinity();
                    if (flat.size(0) > 2) flat[2] = std::numeric_limits<float>::quiet_NaN();
                }
                break;
            case 5:
                // Zero tensor
                input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
        }

        // Test basic cos operation
        auto result1 = torch::cos(input);
        
        // Verify result properties
        if (result1.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape mismatch");
        }
        if (result1.dtype() != input.dtype()) {
            throw std::runtime_error("Output dtype mismatch");
        }
        if (result1.device() != input.device()) {
            throw std::runtime_error("Output device mismatch");
        }

        // Test with output tensor (if we have enough data)
        if (offset < Size) {
            bool use_out_tensor = consume_bool(Data, Size, offset);
            if (use_out_tensor) {
                // Create output tensor with same properties
                auto out_tensor = torch::empty_like(input);
                auto result2 = torch::cos(input, out_tensor);
                
                // Verify that result2 is the same as out_tensor
                if (!torch::equal(result2, out_tensor)) {
                    throw std::runtime_error("Output tensor not properly used");
                }
            }
        }

        // Test mathematical properties where applicable
        if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
            // For finite values, cos should be in [-1, 1]
            auto finite_mask = torch::isfinite(input);
            if (finite_mask.any().item<bool>()) {
                auto finite_result = result1.masked_select(finite_mask);
                auto abs_result = torch::abs(finite_result);
                auto max_val = abs_result.max();
                
                // Allow small numerical errors
                if (max_val.item<double>() > 1.01) {
                    // This might indicate a numerical issue, but don't crash
                    std::cout << "Warning: cos result outside expected range" << std::endl;
                }
            }
        }

        // Test edge cases with specific shapes
        if (offset < Size) {
            uint8_t edge_case = consume_uint8_t(Data, Size, offset);
            switch (edge_case % 4) {
                case 0:
                    // Empty tensor
                    {
                        auto empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype).device(device));
                        auto empty_result = torch::cos(empty_tensor);
                        if (empty_result.numel() != 0) {
                            throw std::runtime_error("Empty tensor result should be empty");
                        }
                    }
                    break;
                case 1:
                    // Scalar tensor
                    {
                        auto scalar_val = consume_float(Data, Size, offset);
                        auto scalar_tensor = torch::tensor(scalar_val, torch::TensorOptions().dtype(dtype).device(device));
                        auto scalar_result = torch::cos(scalar_tensor);
                        if (scalar_result.dim() != 0) {
                            throw std::runtime_error("Scalar result should have 0 dimensions");
                        }
                    }
                    break;
                case 2:
                    // Very large tensor (if memory allows)
                    try {
                        auto large_shape = std::vector<int64_t>{1000, 100};
                        auto large_tensor = torch::randn(large_shape, torch::TensorOptions().dtype(dtype).device(device));
                        auto large_result = torch::cos(large_tensor);
                    } catch (const std::exception&) {
                        // Memory allocation might fail, that's okay
                    }
                    break;
                case 3:
                    // Test with requires_grad if on CPU and floating point
                    if (device.is_cpu() && (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
                        auto grad_tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
                        auto grad_result = torch::cos(grad_tensor);
                        if (!grad_result.requires_grad()) {
                            throw std::runtime_error("Gradient tracking lost");
                        }
                    }
                    break;
            }
        }

        // Force evaluation to catch any lazy evaluation issues
        result1.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
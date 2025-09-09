#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor shape and data type
        auto shape = generateTensorShape(Data, Size, offset);
        if (shape.empty()) {
            return 0; // Skip if no valid shape
        }

        auto dtype = generateTensorDtype(Data, Size, offset);
        
        // Create input tensor with various data patterns
        torch::Tensor input;
        
        // Test different tensor creation strategies
        uint8_t creation_strategy = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 4);
        
        switch (creation_strategy) {
            case 0:
                // Random values
                input = torch::randn(shape, torch::dtype(dtype));
                break;
            case 1:
                // Values around critical points (0, ±1, ±π/4)
                input = torch::randn(shape, torch::dtype(dtype)) * 2.0 - 1.0;
                break;
            case 2:
                // Large values to test numerical stability
                input = torch::randn(shape, torch::dtype(dtype)) * 1000.0;
                break;
            case 3:
                // Small values near zero
                input = torch::randn(shape, torch::dtype(dtype)) * 0.001;
                break;
            case 4:
                // Special values including infinities and edge cases
                input = torch::randn(shape, torch::dtype(dtype));
                if (input.numel() > 0) {
                    auto flat = input.flatten();
                    if (flat.numel() > 0) flat[0] = std::numeric_limits<double>::infinity();
                    if (flat.numel() > 1) flat[1] = -std::numeric_limits<double>::infinity();
                    if (flat.numel() > 2) flat[2] = 0.0;
                    if (flat.numel() > 3) flat[3] = std::numeric_limits<double>::quiet_NaN();
                }
                break;
        }

        // Test in-place vs out-of-place operations
        uint8_t operation_type = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 2);
        
        torch::Tensor result;
        
        switch (operation_type) {
            case 0:
                // Standard atan operation
                result = torch::atan(input);
                break;
            case 1:
                // In-place atan operation
                {
                    torch::Tensor input_copy = input.clone();
                    input_copy.atan_();
                    result = input_copy;
                }
                break;
            case 2:
                // atan with output tensor
                {
                    torch::Tensor output = torch::empty_like(input);
                    torch::atan_out(output, input);
                    result = output;
                }
                break;
        }

        // Verify result properties
        if (result.defined()) {
            // Check that result has same shape as input
            if (!result.sizes().equals(input.sizes())) {
                std::cerr << "Shape mismatch in atan result" << std::endl;
            }
            
            // Check for valid range: atan should return values in (-π/2, π/2)
            if (result.dtype().isFloatingPoint()) {
                auto result_flat = result.flatten();
                for (int64_t i = 0; i < result_flat.numel(); ++i) {
                    double val = result_flat[i].item<double>();
                    if (std::isfinite(val)) {
                        if (val <= -M_PI/2 || val >= M_PI/2) {
                            std::cerr << "atan result out of expected range: " << val << std::endl;
                        }
                    }
                }
            }
        }

        // Test with different tensor properties
        uint8_t tensor_modifier = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 3);
        
        switch (tensor_modifier) {
            case 0:
                // Test with contiguous tensor
                if (!input.is_contiguous()) {
                    input = input.contiguous();
                    result = torch::atan(input);
                }
                break;
            case 1:
                // Test with non-contiguous tensor (transposed)
                if (input.dim() >= 2) {
                    input = input.transpose(0, 1);
                    result = torch::atan(input);
                }
                break;
            case 2:
                // Test with strided tensor
                if (input.numel() > 1) {
                    input = input.flatten()[torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)];
                    result = torch::atan(input);
                }
                break;
            case 3:
                // Test with reshaped tensor
                if (input.numel() > 0) {
                    auto new_shape = std::vector<int64_t>{input.numel()};
                    input = input.reshape(new_shape);
                    result = torch::atan(input);
                }
                break;
        }

        // Test gradient computation if applicable
        if (input.dtype().isFloatingPoint() && offset < Size) {
            uint8_t test_grad = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 1);
            if (test_grad && input.numel() > 0 && input.numel() < 1000) { // Limit size for gradient test
                try {
                    input.requires_grad_(true);
                    auto output = torch::atan(input);
                    auto grad_output = torch::ones_like(output);
                    auto grad_input = torch::autograd::grad({output}, {input}, {grad_output}, 
                                                          /*retain_graph=*/false, 
                                                          /*create_graph=*/false, 
                                                          /*allow_unused=*/true);
                    if (!grad_input.empty() && grad_input[0].defined()) {
                        // Verify gradient is finite where input is finite
                        auto input_finite = torch::isfinite(input);
                        auto grad_finite = torch::isfinite(grad_input[0]);
                        if (!torch::all(torch::logical_or(torch::logical_not(input_finite), grad_finite)).item<bool>()) {
                            std::cerr << "Non-finite gradient for finite input in atan" << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
                    // Gradient computation might fail for some inputs, which is acceptable
                }
            }
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            uint8_t test_cuda = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 1);
            if (test_cuda) {
                try {
                    auto cuda_input = input.to(torch::kCUDA);
                    auto cuda_result = torch::atan(cuda_input);
                    // Move back to CPU for verification
                    cuda_result = cuda_result.to(torch::kCPU);
                } catch (const std::exception& e) {
                    // CUDA operations might fail, which is acceptable
                }
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
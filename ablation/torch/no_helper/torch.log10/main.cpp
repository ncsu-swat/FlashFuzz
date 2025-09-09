#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor shape and dtype
        auto shape = generateTensorShape(Data, Size, offset);
        if (shape.empty()) {
            return 0; // Skip empty shapes
        }
        
        auto dtype = generateDtype(Data, Size, offset);
        
        // Create input tensor with various value ranges to test edge cases
        torch::Tensor input;
        
        // Choose different value generation strategies
        uint8_t strategy = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 6);
        
        switch (strategy) {
            case 0:
                // Positive values (normal case)
                input = torch::rand(shape, torch::dtype(dtype)) + 0.001; // Avoid zero
                break;
            case 1:
                // Values close to 1 (log10(1) = 0)
                input = torch::rand(shape, torch::dtype(dtype)) * 0.1 + 0.95;
                break;
            case 2:
                // Small positive values (large negative log)
                input = torch::rand(shape, torch::dtype(dtype)) * 1e-6 + 1e-8;
                break;
            case 3:
                // Large positive values
                input = torch::rand(shape, torch::dtype(dtype)) * 1e6 + 1e3;
                break;
            case 4:
                // Powers of 10 (should give integer results)
                input = torch::pow(torch::full(shape, 10.0, torch::dtype(dtype)), 
                                 torch::randint(-3, 4, shape, torch::dtype(dtype)));
                break;
            case 5:
                // Mixed positive values including edge cases
                input = torch::rand(shape, torch::dtype(dtype)) * 1000 + 1e-10;
                break;
            case 6:
                // Values that might cause numerical issues
                input = torch::full(shape, 1.0, torch::dtype(dtype));
                if (shape.size() > 0 && torch::numel(input) > 0) {
                    // Set some elements to very small values
                    auto flat = input.flatten();
                    if (flat.size(0) > 0) {
                        flat[0] = std::numeric_limits<float>::min();
                    }
                    if (flat.size(0) > 1) {
                        flat[flat.size(0) - 1] = std::numeric_limits<float>::epsilon();
                    }
                }
                break;
        }
        
        // Ensure input has valid dtype for log10
        if (dtype == torch::kBool || dtype == torch::kByte || dtype == torch::kChar) {
            input = input.to(torch::kFloat32);
        }
        
        // Test basic log10 operation
        torch::Tensor result = torch::log10(input);
        
        // Verify result properties
        if (result.sizes() != input.sizes()) {
            std::cerr << "Output shape mismatch" << std::endl;
        }
        
        // Test with output tensor parameter
        if (consumeBool(Data, Size, offset)) {
            torch::Tensor out = torch::empty_like(result);
            torch::log10_out(out, input);
            
            // Verify out parameter works correctly
            if (!torch::allclose(result, out, 1e-5, 1e-8, /*equal_nan=*/true)) {
                std::cerr << "Output parameter mismatch" << std::endl;
            }
        }
        
        // Test in-place operation if available
        if (consumeBool(Data, Size, offset)) {
            torch::Tensor input_copy = input.clone();
            input_copy.log10_();
            
            if (!torch::allclose(result, input_copy, 1e-5, 1e-8, /*equal_nan=*/true)) {
                std::cerr << "In-place operation mismatch" << std::endl;
            }
        }
        
        // Test edge cases with specific values
        if (consumeBool(Data, Size, offset)) {
            // Test log10(1) = 0
            torch::Tensor ones = torch::ones({2, 2}, torch::dtype(input.dtype()));
            torch::Tensor log_ones = torch::log10(ones);
            
            // Test log10(10) = 1
            torch::Tensor tens = torch::full({2, 2}, 10.0, torch::dtype(input.dtype()));
            torch::Tensor log_tens = torch::log10(tens);
            
            // Test log10(0.1) = -1
            torch::Tensor point_ones = torch::full({2, 2}, 0.1, torch::dtype(input.dtype()));
            torch::Tensor log_point_ones = torch::log10(point_ones);
        }
        
        // Test with different tensor layouts
        if (input.dim() >= 2 && consumeBool(Data, Size, offset)) {
            torch::Tensor transposed = input.transpose(0, 1);
            torch::Tensor result_transposed = torch::log10(transposed);
        }
        
        // Test with non-contiguous tensors
        if (input.dim() >= 1 && input.size(0) > 1 && consumeBool(Data, Size, offset)) {
            torch::Tensor sliced = input.slice(0, 0, input.size(0), 2);
            torch::Tensor result_sliced = torch::log10(sliced);
        }
        
        // Test gradient computation if tensor requires grad
        if (input.dtype().isFloatingPoint() && consumeBool(Data, Size, offset)) {
            torch::Tensor grad_input = input.clone().detach().requires_grad_(true);
            torch::Tensor grad_result = torch::log10(grad_input);
            
            if (torch::numel(grad_result) > 0) {
                torch::Tensor grad_output = torch::ones_like(grad_result);
                auto grads = torch::autograd::grad({grad_result}, {grad_input}, {grad_output}, 
                                                 /*retain_graph=*/false, /*create_graph=*/false, 
                                                 /*allow_unused=*/true);
                if (!grads.empty() && grads[0].defined()) {
                    // Gradient should be 1/(x * ln(10))
                    torch::Tensor expected_grad = 1.0 / (grad_input * std::log(10.0));
                    if (!torch::allclose(grads[0], expected_grad, 1e-4, 1e-6, /*equal_nan=*/true)) {
                        std::cerr << "Gradient computation mismatch" << std::endl;
                    }
                }
            }
        }
        
        // Force evaluation of all tensors to catch any lazy evaluation issues
        result.sum().item<double>();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
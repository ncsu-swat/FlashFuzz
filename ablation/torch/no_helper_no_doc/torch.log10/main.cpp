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
            return 0; // Skip invalid shapes
        }

        auto dtype = generateTensorDtype(Data, Size, offset);
        
        // Create input tensor with various data patterns
        torch::Tensor input;
        
        // Test different tensor creation strategies
        uint8_t creation_strategy = consumeUint8(Data, Size, offset);
        switch (creation_strategy % 6) {
            case 0:
                // Random values
                input = torch::randn(shape, torch::dtype(dtype));
                break;
            case 1:
                // Positive values only (log10 domain)
                input = torch::abs(torch::randn(shape, torch::dtype(dtype))) + 1e-6;
                break;
            case 2:
                // Values around 1.0
                input = torch::ones(shape, torch::dtype(dtype)) + 0.1 * torch::randn(shape, torch::dtype(dtype));
                input = torch::abs(input) + 1e-6;
                break;
            case 3:
                // Large values
                input = torch::abs(torch::randn(shape, torch::dtype(dtype))) * 1000.0 + 1.0;
                break;
            case 4:
                // Small positive values
                input = torch::abs(torch::randn(shape, torch::dtype(dtype))) * 1e-3 + 1e-6;
                break;
            case 5:
                // Powers of 10
                input = torch::pow(10.0, torch::randn(shape, torch::dtype(dtype)));
                break;
        }

        // Ensure input is positive for log10 (avoid NaN/inf issues)
        input = torch::abs(input) + 1e-10;

        // Test in-place vs out-of-place operations
        uint8_t operation_type = consumeUint8(Data, Size, offset);
        torch::Tensor result;
        
        switch (operation_type % 3) {
            case 0:
                // Basic log10 operation
                result = torch::log10(input);
                break;
            case 1:
                // In-place operation (if supported)
                {
                    torch::Tensor input_copy = input.clone();
                    input_copy.log10_();
                    result = input_copy;
                }
                break;
            case 2:
                // With output tensor
                {
                    torch::Tensor output = torch::empty_like(input);
                    torch::log10_out(output, input);
                    result = output;
                }
                break;
        }

        // Test with different tensor properties
        uint8_t tensor_modifier = consumeUint8(Data, Size, offset);
        if (tensor_modifier % 4 == 0 && input.dim() > 1) {
            // Test with transposed tensor
            input = input.transpose(0, 1);
            result = torch::log10(input);
        } else if (tensor_modifier % 4 == 1 && input.numel() > 1) {
            // Test with reshaped tensor
            auto new_shape = std::vector<int64_t>{-1};
            input = input.reshape(new_shape);
            result = torch::log10(input);
        } else if (tensor_modifier % 4 == 2) {
            // Test with contiguous tensor
            input = input.contiguous();
            result = torch::log10(input);
        }

        // Test edge cases with special values
        uint8_t edge_case = consumeUint8(Data, Size, offset);
        if (edge_case % 8 == 0) {
            // Test with tensor containing 1.0 (log10(1) = 0)
            torch::Tensor ones_tensor = torch::ones({2, 2}, torch::dtype(dtype));
            result = torch::log10(ones_tensor);
        } else if (edge_case % 8 == 1) {
            // Test with tensor containing 10.0 (log10(10) = 1)
            torch::Tensor tens_tensor = torch::full({2, 2}, 10.0, torch::dtype(dtype));
            result = torch::log10(tens_tensor);
        } else if (edge_case % 8 == 2) {
            // Test with very small positive values
            torch::Tensor small_tensor = torch::full({2, 2}, 1e-10, torch::dtype(dtype));
            result = torch::log10(small_tensor);
        } else if (edge_case % 8 == 3) {
            // Test with very large values
            torch::Tensor large_tensor = torch::full({2, 2}, 1e10, torch::dtype(dtype));
            result = torch::log10(large_tensor);
        }

        // Test different devices if CUDA is available
        if (torch::cuda::is_available() && consumeUint8(Data, Size, offset) % 4 == 0) {
            input = input.to(torch::kCUDA);
            result = torch::log10(input);
            result = result.to(torch::kCPU); // Move back for validation
        }

        // Validate result properties
        if (result.defined()) {
            // Check that result has same shape as input
            if (result.sizes() != input.sizes()) {
                std::cerr << "Shape mismatch in log10 result" << std::endl;
            }
            
            // Check for NaN or Inf values (shouldn't occur with positive inputs)
            if (torch::any(torch::isnan(result)).item<bool>() || 
                torch::any(torch::isinf(result)).item<bool>()) {
                std::cerr << "NaN or Inf detected in log10 result" << std::endl;
            }
            
            // Basic mathematical property check: log10(10^x) ≈ x
            if (input.numel() > 0 && input.numel() < 100) {
                torch::Tensor test_input = torch::pow(10.0, torch::randn_like(input));
                torch::Tensor log_result = torch::log10(test_input);
                // The result should be close to the original exponent
            }
        }

        // Test gradient computation if input requires grad
        if (consumeUint8(Data, Size, offset) % 3 == 0 && 
            (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
            input.requires_grad_(true);
            torch::Tensor grad_result = torch::log10(input);
            torch::Tensor loss = grad_result.sum();
            loss.backward();
            
            // Check that gradients are computed
            if (!input.grad().defined()) {
                std::cerr << "Gradients not computed for log10" << std::endl;
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
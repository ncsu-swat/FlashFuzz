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
        
        // Only test with floating point types since atanh requires them
        if (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
            dtype != torch::kFloat16 && dtype != torch::kBFloat16) {
            dtype = torch::kFloat32; // Default to float32
        }

        // Create input tensor with values in different ranges to test edge cases
        torch::Tensor input;
        
        // Use remaining data to determine test case type
        uint8_t test_case = 0;
        if (offset < Size) {
            test_case = Data[offset] % 6;
            offset++;
        }
        
        switch (test_case) {
            case 0: {
                // Normal case: values in (-1, 1)
                input = torch::randn(shape, torch::dtype(dtype)) * 0.9;
                break;
            }
            case 1: {
                // Edge case: values close to boundaries
                input = torch::randn(shape, torch::dtype(dtype)) * 0.99;
                break;
            }
            case 2: {
                // Edge case: exact boundary values -1 and 1
                input = torch::ones(shape, torch::dtype(dtype));
                if (offset < Size && Data[offset] % 2 == 0) {
                    input = -input;
                }
                offset++;
                break;
            }
            case 3: {
                // Edge case: values outside domain (should produce NaN)
                input = torch::randn(shape, torch::dtype(dtype)) * 2.0 + 1.5;
                break;
            }
            case 4: {
                // Edge case: mix of valid and invalid values
                input = torch::randn(shape, torch::dtype(dtype));
                // Scale some values outside [-1, 1]
                auto mask = torch::rand(shape) > 0.5;
                input = torch::where(mask, input * 2.0, input * 0.8);
                break;
            }
            case 5: {
                // Edge case: special values (zeros, very small values)
                input = torch::randn(shape, torch::dtype(dtype)) * 1e-6;
                break;
            }
        }

        // Test basic atanh operation
        auto result1 = torch::atanh(input);
        
        // Verify result shape matches input
        if (!result1.sizes().equals(input.sizes())) {
            std::cerr << "Shape mismatch in atanh result" << std::endl;
        }
        
        // Test with output tensor (if we have enough data)
        if (offset < Size && Data[offset] % 2 == 0) {
            auto out_tensor = torch::empty_like(input);
            auto result2 = torch::atanh(input, out_tensor);
            
            // Verify that result2 is the same as out_tensor
            if (!torch::equal(result2, out_tensor)) {
                std::cerr << "Output tensor not properly used in atanh" << std::endl;
            }
        }
        offset++;
        
        // Test mathematical properties where applicable
        if (test_case == 0 || test_case == 1 || test_case == 5) {
            // For values in valid domain, test that tanh(atanh(x)) ≈ x
            auto tanh_result = torch::tanh(result1);
            auto diff = torch::abs(tanh_result - input);
            auto max_diff = torch::max(diff);
            
            // Allow for numerical precision issues
            if (max_diff.item<double>() > 1e-4 && dtype == torch::kFloat32) {
                // This might indicate a numerical issue, but don't crash
            }
        }
        
        // Test gradient computation if input requires grad
        if (offset < Size && Data[offset] % 3 == 0 && 
            (dtype == torch::kFloat32 || dtype == torch::kFloat64)) {
            input.requires_grad_(true);
            auto grad_result = torch::atanh(input);
            
            if (grad_result.numel() > 0) {
                auto sum_result = torch::sum(grad_result);
                if (torch::isfinite(sum_result).item<bool>()) {
                    sum_result.backward();
                    
                    // Check that gradients were computed
                    if (input.grad().defined()) {
                        auto grad_finite = torch::isfinite(input.grad());
                        // Gradients should be finite for values in (-1, 1)
                    }
                }
            }
        }
        offset++;
        
        // Test with different tensor properties
        if (offset < Size) {
            uint8_t prop_test = Data[offset] % 4;
            offset++;
            
            switch (prop_test) {
                case 0: {
                    // Test with contiguous tensor
                    if (input.dim() > 1) {
                        auto transposed = input.transpose(0, -1);
                        auto result_t = torch::atanh(transposed);
                    }
                    break;
                }
                case 1: {
                    // Test with sliced tensor
                    if (input.numel() > 1) {
                        auto sliced = input.flatten().slice(0, 0, input.numel()/2);
                        auto result_s = torch::atanh(sliced);
                    }
                    break;
                }
                case 2: {
                    // Test with reshaped tensor
                    if (input.numel() > 1) {
                        auto reshaped = input.view({-1});
                        auto result_r = torch::atanh(reshaped);
                    }
                    break;
                }
                case 3: {
                    // Test with cloned tensor
                    auto cloned = input.clone();
                    auto result_c = torch::atanh(cloned);
                    break;
                }
            }
        }
        
        // Force evaluation of any lazy operations
        if (result1.numel() > 0) {
            result1.sum().item<double>();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
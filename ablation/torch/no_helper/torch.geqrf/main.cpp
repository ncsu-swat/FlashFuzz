#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions
        auto dims = parseTensorDimensions(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (dims.empty()) return 0;

        // Parse data type
        auto dtype = parseDType(Data, Size, offset);
        if (!dtype.has_value()) return 0;

        // Only test with floating point types as geqrf requires them
        if (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
            dtype != torch::kComplexFloat && dtype != torch::kComplexDouble) {
            return 0;
        }

        // Parse device
        auto device = parseDevice(Data, Size, offset);
        if (!device.has_value()) return 0;

        // Create input tensor with parsed parameters
        torch::Tensor input;
        try {
            input = createTensor(dims, dtype.value(), device.value(), Data, Size, offset);
        } catch (...) {
            return 0;
        }

        // Ensure we have at least a 2D tensor for QR decomposition
        if (input.dim() < 2) {
            return 0;
        }

        // Test basic geqrf call
        auto result = torch::geqrf(input);
        auto a = std::get<0>(result);
        auto tau = std::get<1>(result);

        // Verify output shapes
        if (a.sizes() != input.sizes()) {
            std::cerr << "geqrf output 'a' has incorrect shape" << std::endl;
        }

        // tau should have shape (..., min(m, n)) where input is (..., m, n)
        auto input_sizes = input.sizes().vec();
        auto expected_tau_sizes = input_sizes;
        int m = input_sizes[input_sizes.size() - 2];
        int n = input_sizes[input_sizes.size() - 1];
        expected_tau_sizes[expected_tau_sizes.size() - 1] = std::min(m, n);
        expected_tau_sizes.pop_back(); // Remove the last dimension, then add min(m,n)
        expected_tau_sizes.push_back(std::min(m, n));

        if (tau.sizes().vec() != expected_tau_sizes) {
            std::cerr << "geqrf output 'tau' has incorrect shape" << std::endl;
        }

        // Test with output tensors pre-allocated
        if (parseFlag(Data, Size, offset)) {
            try {
                torch::Tensor out_a = torch::empty_like(input);
                torch::Tensor out_tau = torch::empty(expected_tau_sizes, input.options());
                
                auto result_with_out = torch::geqrf(input, std::make_tuple(out_a, out_tau));
                
                // Verify that the output tensors were used
                if (!torch::allclose(std::get<0>(result_with_out), out_a, 1e-5, 1e-8, /*equal_nan=*/true)) {
                    std::cerr << "geqrf with pre-allocated output 'a' mismatch" << std::endl;
                }
                if (!torch::allclose(std::get<1>(result_with_out), out_tau, 1e-5, 1e-8, /*equal_nan=*/true)) {
                    std::cerr << "geqrf with pre-allocated output 'tau' mismatch" << std::endl;
                }
            } catch (...) {
                // Pre-allocated output might fail for various reasons, continue testing
            }
        }

        // Test edge cases
        if (parseFlag(Data, Size, offset)) {
            // Test with very small matrices
            if (input.size(-2) >= 1 && input.size(-1) >= 1) {
                try {
                    auto small_input = input.narrow(-2, 0, 1).narrow(-1, 0, 1);
                    auto small_result = torch::geqrf(small_input);
                } catch (...) {
                    // Small matrices might have numerical issues
                }
            }
        }

        // Test with different memory layouts if possible
        if (parseFlag(Data, Size, offset) && input.dim() == 2) {
            try {
                auto transposed = input.transpose(-2, -1).contiguous();
                auto transposed_result = torch::geqrf(transposed);
            } catch (...) {
                // Transposed input might fail for non-square matrices
            }
        }

        // Test gradient computation if input requires grad
        if (parseFlag(Data, Size, offset) && input.dtype().is_floating_point()) {
            try {
                auto input_with_grad = input.detach().requires_grad_(true);
                auto grad_result = torch::geqrf(input_with_grad);
                
                // Try to compute gradients
                auto loss = std::get<0>(grad_result).sum() + std::get<1>(grad_result).sum();
                loss.backward();
                
                if (!input_with_grad.grad().defined()) {
                    std::cerr << "geqrf gradient computation failed" << std::endl;
                }
            } catch (...) {
                // Gradient computation might fail for singular matrices
            }
        }

        // Verify numerical properties when possible
        if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
            try {
                // Check that the result is finite
                if (!torch::all(torch::isfinite(a)).item<bool>()) {
                    std::cerr << "geqrf produced non-finite values in 'a'" << std::endl;
                }
                if (!torch::all(torch::isfinite(tau)).item<bool>()) {
                    std::cerr << "geqrf produced non-finite values in 'tau'" << std::endl;
                }
            } catch (...) {
                // Numerical checks might fail
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
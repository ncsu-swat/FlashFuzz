#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor parameters
        auto tensor_params = generateTensorParams(Data, Size, offset);
        if (tensor_params.empty()) return 0;

        // Create input tensor with various data types
        auto dtype = generateDtype(Data, Size, offset);
        if (!isFloatingPointType(dtype)) {
            dtype = torch::kFloat32; // nanmedian requires floating point types
        }

        auto input = createTensor(tensor_params[0], dtype);
        if (!input.defined() || input.numel() == 0) return 0;

        // Introduce NaN values randomly
        if (consumeBool(Data, Size, offset)) {
            auto flat_input = input.flatten();
            int64_t num_elements = flat_input.numel();
            
            // Add NaN values at random positions
            int num_nans = consumeIntegralInRange<int>(Data, Size, offset, 0, std::min(num_elements, 10L));
            for (int i = 0; i < num_nans; i++) {
                int64_t pos = consumeIntegralInRange<int64_t>(Data, Size, offset, 0, num_elements - 1);
                flat_input[pos] = std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Test case 1: Basic nanmedian without dimension
        auto result1 = torch::nanmedian(input);
        
        // Verify result is valid
        if (result1.defined()) {
            auto result_val = result1.item<double>();
            // Check if result is finite or NaN (both are valid)
            (void)std::isfinite(result_val); // Just access to trigger any issues
        }

        // Test case 2: nanmedian with dimension parameter
        if (input.dim() > 0) {
            int64_t dim = consumeIntegralInRange<int64_t>(Data, Size, offset, -input.dim(), input.dim() - 1);
            bool keepdim = consumeBool(Data, Size, offset);
            
            auto result2 = torch::nanmedian(input, dim, keepdim);
            auto values = std::get<0>(result2);
            auto indices = std::get<1>(result2);
            
            // Verify results are valid
            if (values.defined() && indices.defined()) {
                // Check that indices have correct dtype
                if (indices.dtype() != torch::kLong) {
                    throw std::runtime_error("Indices should have Long dtype");
                }
                
                // Check dimensions
                if (keepdim) {
                    if (values.dim() != input.dim()) {
                        throw std::runtime_error("keepdim=True should preserve dimensions");
                    }
                } else {
                    if (input.dim() > 1 && values.dim() != input.dim() - 1) {
                        throw std::runtime_error("keepdim=False should reduce dimensions");
                    }
                }
            }
        }

        // Test case 3: Edge cases with all NaN tensor
        if (consumeBool(Data, Size, offset)) {
            auto all_nan_tensor = torch::full_like(input, std::numeric_limits<double>::quiet_NaN());
            auto result3 = torch::nanmedian(all_nan_tensor);
            
            if (result3.defined()) {
                auto val = result3.item<double>();
                if (!std::isnan(val)) {
                    throw std::runtime_error("All NaN tensor should return NaN");
                }
            }
        }

        // Test case 4: Single element tensor
        if (consumeBool(Data, Size, offset)) {
            auto single_val = consumeFloatingPoint<double>(Data, Size, offset);
            auto single_tensor = torch::tensor({single_val}, dtype);
            auto result4 = torch::nanmedian(single_tensor);
            
            if (result4.defined()) {
                auto val = result4.item<double>();
                if (std::isnan(single_val)) {
                    if (!std::isnan(val)) {
                        throw std::runtime_error("Single NaN should return NaN");
                    }
                } else {
                    if (std::abs(val - single_val) > 1e-6) {
                        throw std::runtime_error("Single element median should equal the element");
                    }
                }
            }
        }

        // Test case 5: Test with output tensors
        if (input.dim() > 0 && consumeBool(Data, Size, offset)) {
            int64_t dim = consumeIntegralInRange<int64_t>(Data, Size, offset, -input.dim(), input.dim() - 1);
            bool keepdim = consumeBool(Data, Size, offset);
            
            // Calculate expected output shape
            auto expected_shape = input.sizes().vec();
            if (keepdim) {
                expected_shape[dim < 0 ? dim + input.dim() : dim] = 1;
            } else {
                expected_shape.erase(expected_shape.begin() + (dim < 0 ? dim + input.dim() : dim));
            }
            
            auto out_values = torch::empty(expected_shape, input.options());
            auto out_indices = torch::empty(expected_shape, torch::kLong);
            
            torch::nanmedian_out(out_values, out_indices, input, dim, keepdim);
            
            // Verify outputs are properly filled
            if (!out_values.defined() || !out_indices.defined()) {
                throw std::runtime_error("Output tensors should be defined after nanmedian_out");
            }
        }

        // Test case 6: Different tensor shapes and dimensions
        if (consumeBool(Data, Size, offset) && input.dim() > 1) {
            // Test with different dimensions
            for (int64_t d = 0; d < input.dim(); d++) {
                if (input.size(d) > 1) {
                    auto result_dim = torch::nanmedian(input, d, false);
                    auto values = std::get<0>(result_dim);
                    auto indices = std::get<1>(result_dim);
                    
                    // Basic validation
                    if (values.defined() && indices.defined()) {
                        if (indices.dtype() != torch::kLong) {
                            throw std::runtime_error("Indices dtype should be Long");
                        }
                    }
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
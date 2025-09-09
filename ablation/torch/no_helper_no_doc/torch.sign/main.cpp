#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor shape and data type
        auto shape = parse_shape(Data, Size, offset);
        if (shape.empty()) {
            return 0; // Invalid shape, discard input
        }

        auto dtype = parse_dtype(Data, Size, offset);
        if (dtype == torch::kUndefined) {
            return 0; // Invalid dtype, discard input
        }

        // Create input tensor with various edge cases
        torch::Tensor input;
        
        // Try different tensor creation strategies based on remaining data
        if (offset < Size) {
            uint8_t strategy = Data[offset++];
            
            switch (strategy % 6) {
                case 0: {
                    // Create tensor from parsed data
                    input = create_tensor_from_data(Data, Size, offset, shape, dtype);
                    break;
                }
                case 1: {
                    // Create tensor with zeros
                    input = torch::zeros(shape, dtype);
                    break;
                }
                case 2: {
                    // Create tensor with ones
                    input = torch::ones(shape, dtype);
                    break;
                }
                case 3: {
                    // Create tensor with random values
                    input = torch::randn(shape, dtype);
                    break;
                }
                case 4: {
                    // Create tensor with extreme values
                    input = torch::full(shape, std::numeric_limits<double>::infinity(), dtype);
                    break;
                }
                case 5: {
                    // Create tensor with negative extreme values
                    input = torch::full(shape, -std::numeric_limits<double>::infinity(), dtype);
                    break;
                }
            }
        } else {
            // Default case: create random tensor
            input = torch::randn(shape, dtype);
        }

        // Add special values to test edge cases
        if (input.numel() > 0 && offset < Size) {
            auto flat_input = input.flatten();
            int64_t num_elements = flat_input.numel();
            
            // Inject special values based on remaining data
            for (size_t i = offset; i < Size && (i - offset) < static_cast<size_t>(num_elements); ++i) {
                int64_t idx = (i - offset) % num_elements;
                uint8_t value_type = Data[i];
                
                switch (value_type % 8) {
                    case 0:
                        flat_input[idx] = 0.0; // Zero
                        break;
                    case 1:
                        flat_input[idx] = 1.0; // Positive
                        break;
                    case 2:
                        flat_input[idx] = -1.0; // Negative
                        break;
                    case 3:
                        flat_input[idx] = std::numeric_limits<double>::infinity(); // +inf
                        break;
                    case 4:
                        flat_input[idx] = -std::numeric_limits<double>::infinity(); // -inf
                        break;
                    case 5:
                        flat_input[idx] = std::numeric_limits<double>::quiet_NaN(); // NaN
                        break;
                    case 6:
                        flat_input[idx] = std::numeric_limits<double>::epsilon(); // Very small positive
                        break;
                    case 7:
                        flat_input[idx] = -std::numeric_limits<double>::epsilon(); // Very small negative
                        break;
                }
            }
            
            input = flat_input.reshape(shape);
        }

        // Test torch.sign with the input tensor
        torch::Tensor result = torch::sign(input);

        // Verify result properties
        if (result.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape mismatch");
        }

        // Test in-place version if applicable
        if (input.dtype().isFloatingPoint() || input.dtype().isIntegral()) {
            torch::Tensor input_copy = input.clone();
            torch::Tensor inplace_result = torch::sign_(input_copy);
            
            // Verify in-place operation returns the same tensor
            if (!torch::equal(inplace_result, input_copy)) {
                throw std::runtime_error("In-place operation inconsistency");
            }
        }

        // Test with different tensor properties
        if (input.numel() > 0) {
            // Test with contiguous tensor
            if (!input.is_contiguous()) {
                torch::Tensor contiguous_input = input.contiguous();
                torch::Tensor contiguous_result = torch::sign(contiguous_input);
            }

            // Test with non-contiguous tensor (if possible)
            if (input.dim() > 1) {
                torch::Tensor transposed = input.transpose(0, -1);
                torch::Tensor transposed_result = torch::sign(transposed);
            }

            // Test with different memory formats if tensor is 4D
            if (input.dim() == 4 && input.size(1) > 1) {
                try {
                    torch::Tensor channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                    torch::Tensor channels_last_result = torch::sign(channels_last);
                } catch (...) {
                    // Channels last might not be supported for all dtypes
                }
            }
        }

        // Test gradient computation if input requires grad
        if (input.dtype().isFloatingPoint()) {
            torch::Tensor grad_input = input.clone().requires_grad_(true);
            torch::Tensor grad_output = torch::sign(grad_input);
            
            if (grad_output.numel() > 0) {
                try {
                    torch::Tensor grad_sum = grad_output.sum();
                    grad_sum.backward();
                } catch (...) {
                    // Gradient computation might fail for some edge cases
                }
            }
        }

        // Additional edge case: empty tensor
        torch::Tensor empty_tensor = torch::empty({0}, dtype);
        torch::Tensor empty_result = torch::sign(empty_tensor);

        // Test with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(42.0, dtype);
        torch::Tensor scalar_result = torch::sign(scalar_tensor);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
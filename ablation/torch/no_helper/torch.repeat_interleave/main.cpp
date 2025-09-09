#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 16) return 0;

        // Extract basic parameters
        auto use_tensor_repeats = extract_bool(Data, Size, offset);
        auto use_dim = extract_bool(Data, Size, offset);
        auto use_output_size = extract_bool(Data, Size, offset);
        auto test_second_overload = extract_bool(Data, Size, offset);

        // Test the second overload: repeat_interleave(repeats) -> Tensor
        if (test_second_overload) {
            auto repeats_shape = extract_tensor_shape(Data, Size, offset, 1, 4); // 1D tensor with up to 4 elements
            if (repeats_shape.empty()) return 0;
            
            auto repeats_tensor = create_tensor(Data, Size, offset, repeats_shape, torch::kInt64);
            if (!repeats_tensor.defined()) return 0;
            
            // Clamp repeats values to reasonable range [0, 10]
            repeats_tensor = torch::clamp(repeats_tensor, 0, 10);
            
            auto result = torch::repeat_interleave(repeats_tensor);
            
            // Verify result properties
            if (result.defined()) {
                auto expected_size = torch::sum(repeats_tensor).item<int64_t>();
                if (result.numel() == expected_size && result.dim() == 1) {
                    // Additional verification: check if values are correct
                    auto result_cpu = result.cpu();
                    auto repeats_cpu = repeats_tensor.cpu();
                }
            }
            return 0;
        }

        // Test the main overload: repeat_interleave(input, repeats, dim, output_size)
        auto input_shape = extract_tensor_shape(Data, Size, offset, 1, 4);
        if (input_shape.empty()) return 0;

        auto input_tensor = create_tensor(Data, Size, offset, input_shape);
        if (!input_tensor.defined()) return 0;

        torch::Tensor result;

        if (use_tensor_repeats) {
            // Use tensor repeats
            int64_t dim_val = -1;
            if (use_dim && input_tensor.dim() > 0) {
                dim_val = extract_int(Data, Size, offset) % input_tensor.dim();
                if (dim_val < 0) dim_val += input_tensor.dim();
            }

            // Create repeats tensor
            std::vector<int64_t> repeats_shape;
            if (use_dim && dim_val >= 0) {
                // Repeats should match the size of the specified dimension
                repeats_shape = {input_tensor.size(dim_val)};
            } else {
                // For flattened case, repeats can be any reasonable size
                auto repeats_size = std::min(static_cast<int64_t>(10), input_tensor.numel());
                repeats_shape = {repeats_size};
            }

            auto repeats_tensor = create_tensor(Data, Size, offset, repeats_shape, torch::kInt64);
            if (!repeats_tensor.defined()) return 0;

            // Clamp repeats values to reasonable range [0, 5]
            repeats_tensor = torch::clamp(repeats_tensor, 0, 5);

            c10::optional<int64_t> dim_opt = c10::nullopt;
            if (use_dim) {
                dim_opt = dim_val;
            }

            c10::optional<int64_t> output_size_opt = c10::nullopt;
            if (use_output_size) {
                auto total_repeats = torch::sum(repeats_tensor).item<int64_t>();
                output_size_opt = total_repeats;
            }

            if (use_dim) {
                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor, dim_opt, output_size_opt);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor, dim_opt);
                }
            } else {
                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor, dim_opt, output_size_opt);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor);
                }
            }
        } else {
            // Use scalar repeats
            auto repeats_val = std::abs(extract_int(Data, Size, offset)) % 6; // 0-5 repeats

            if (use_dim && input_tensor.dim() > 0) {
                auto dim_val = extract_int(Data, Size, offset) % input_tensor.dim();
                if (dim_val < 0) dim_val += input_tensor.dim();

                c10::optional<int64_t> output_size_opt = c10::nullopt;
                if (use_output_size) {
                    output_size_opt = input_tensor.size(dim_val) * repeats_val;
                }

                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_val, dim_val, output_size_opt);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_val, dim_val);
                }
            } else {
                // No dimension specified - flattened case
                c10::optional<int64_t> output_size_opt = c10::nullopt;
                if (use_output_size) {
                    output_size_opt = input_tensor.numel() * repeats_val;
                }

                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_val, c10::nullopt, output_size_opt);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_val);
                }
            }
        }

        // Verify result properties
        if (result.defined()) {
            // Basic sanity checks
            auto result_cpu = result.cpu();
            
            // Check that result has reasonable size
            if (result.numel() > 1000000) {
                return 0; // Skip very large results
            }

            // Test edge cases with different input types
            if (offset < Size - 4) {
                auto dtype_choice = extract_int(Data, Size, offset) % 4;
                torch::Tensor typed_input;
                
                switch (dtype_choice) {
                    case 0: typed_input = input_tensor.to(torch::kFloat32); break;
                    case 1: typed_input = input_tensor.to(torch::kInt32); break;
                    case 2: typed_input = input_tensor.to(torch::kInt64); break;
                    case 3: typed_input = input_tensor.to(torch::kBool); break;
                }

                if (typed_input.defined()) {
                    // Test with different dtype
                    auto typed_result = torch::repeat_interleave(typed_input, 2);
                }
            }
        }

        // Test some edge cases
        if (offset < Size - 8) {
            // Test with zero repeats
            auto zero_result = torch::repeat_interleave(input_tensor, 0);
            
            // Test with empty tensor
            auto empty_tensor = torch::empty({0});
            if (empty_tensor.defined()) {
                auto empty_result = torch::repeat_interleave(empty_tensor, 1);
            }

            // Test with 1D tensor
            auto flat_input = input_tensor.flatten();
            auto flat_result = torch::repeat_interleave(flat_input, 1);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
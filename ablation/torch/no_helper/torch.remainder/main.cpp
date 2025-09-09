#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Generate input tensor
        auto input_info = generate_tensor_info(Data, Size, offset);
        if (input_info.numel == 0) {
            return 0;
        }

        // Create input tensor with various dtypes (excluding complex)
        torch::Tensor input;
        auto dtype_choice = get_value<uint8_t>(Data, Size, offset) % 6;
        switch (dtype_choice) {
            case 0: input = create_tensor(input_info, torch::kFloat32); break;
            case 1: input = create_tensor(input_info, torch::kFloat64); break;
            case 2: input = create_tensor(input_info, torch::kInt32); break;
            case 3: input = create_tensor(input_info, torch::kInt64); break;
            case 4: input = create_tensor(input_info, torch::kInt8); break;
            case 5: input = create_tensor(input_info, torch::kInt16); break;
        }

        // Test different types of divisors
        auto divisor_type = get_value<uint8_t>(Data, Size, offset) % 4;
        
        if (divisor_type == 0) {
            // Test with scalar divisor
            auto scalar_val = get_value<float>(Data, Size, offset);
            
            // Avoid zero divisor for most cases, but test it occasionally
            if (scalar_val == 0.0f) {
                auto zero_test = get_value<uint8_t>(Data, Size, offset) % 10;
                if (zero_test != 0) {
                    scalar_val = 1.0f; // Replace zero with non-zero
                }
            }
            
            // Test with positive and negative scalars
            if (get_value<uint8_t>(Data, Size, offset) % 2 == 0) {
                scalar_val = -std::abs(scalar_val);
            }
            
            auto result = torch::remainder(input, scalar_val);
            
            // Test with out parameter
            if (get_value<uint8_t>(Data, Size, offset) % 3 == 0) {
                auto out_tensor = torch::empty_like(result);
                torch::remainder_out(out_tensor, input, scalar_val);
            }
            
        } else if (divisor_type == 1) {
            // Test with tensor divisor of same shape
            auto other_info = input_info; // Same shape as input
            torch::Tensor other;
            
            auto other_dtype_choice = get_value<uint8_t>(Data, Size, offset) % 6;
            switch (other_dtype_choice) {
                case 0: other = create_tensor(other_info, torch::kFloat32); break;
                case 1: other = create_tensor(other_info, torch::kFloat64); break;
                case 2: other = create_tensor(other_info, torch::kInt32); break;
                case 3: other = create_tensor(other_info, torch::kInt64); break;
                case 4: other = create_tensor(other_info, torch::kInt8); break;
                case 5: other = create_tensor(other_info, torch::kInt16); break;
            }
            
            // Avoid zero divisors in most cases
            auto mask = other == 0;
            if (mask.any().item<bool>()) {
                auto zero_replacement = get_value<uint8_t>(Data, Size, offset) % 10;
                if (zero_replacement != 0) {
                    other = torch::where(mask, torch::ones_like(other), other);
                }
            }
            
            auto result = torch::remainder(input, other);
            
            // Test with out parameter
            if (get_value<uint8_t>(Data, Size, offset) % 3 == 0) {
                auto out_tensor = torch::empty_like(result);
                torch::remainder_out(out_tensor, input, other);
            }
            
        } else if (divisor_type == 2) {
            // Test with broadcasting - different shapes
            auto other_info = generate_tensor_info(Data, Size, offset);
            if (other_info.numel > 0) {
                torch::Tensor other = create_tensor(other_info, torch::kFloat32);
                
                // Avoid zero divisors
                auto mask = other == 0;
                if (mask.any().item<bool>()) {
                    other = torch::where(mask, torch::ones_like(other), other);
                }
                
                auto result = torch::remainder(input, other);
            }
            
        } else {
            // Test edge cases with special values
            auto special_case = get_value<uint8_t>(Data, Size, offset) % 5;
            
            switch (special_case) {
                case 0: {
                    // Test with very small divisor
                    auto small_val = 1e-6f;
                    if (get_value<uint8_t>(Data, Size, offset) % 2 == 0) {
                        small_val = -small_val;
                    }
                    auto result = torch::remainder(input, small_val);
                    break;
                }
                case 1: {
                    // Test with very large divisor
                    auto large_val = 1e6f;
                    if (get_value<uint8_t>(Data, Size, offset) % 2 == 0) {
                        large_val = -large_val;
                    }
                    auto result = torch::remainder(input, large_val);
                    break;
                }
                case 2: {
                    // Test with fractional divisor
                    auto frac_val = 0.5f + get_value<float>(Data, Size, offset) * 0.001f;
                    if (get_value<uint8_t>(Data, Size, offset) % 2 == 0) {
                        frac_val = -frac_val;
                    }
                    auto result = torch::remainder(input, frac_val);
                    break;
                }
                case 3: {
                    // Test with integer divisor
                    auto int_val = static_cast<int>(get_value<uint8_t>(Data, Size, offset) % 10 + 1);
                    if (get_value<uint8_t>(Data, Size, offset) % 2 == 0) {
                        int_val = -int_val;
                    }
                    auto result = torch::remainder(input, int_val);
                    break;
                }
                case 4: {
                    // Test with tensor containing mixed positive/negative values
                    auto mixed_tensor = torch::randn_like(input.to(torch::kFloat32));
                    // Ensure no zeros
                    mixed_tensor = torch::where(mixed_tensor == 0, torch::ones_like(mixed_tensor), mixed_tensor);
                    auto result = torch::remainder(input, mixed_tensor);
                    break;
                }
            }
        }

        // Test method call syntax as well
        if (get_value<uint8_t>(Data, Size, offset) % 4 == 0) {
            auto divisor_val = get_value<float>(Data, Size, offset);
            if (divisor_val == 0.0f) {
                divisor_val = 1.0f;
            }
            auto result = input.remainder(divisor_val);
        }

        // Test in-place operation
        if (get_value<uint8_t>(Data, Size, offset) % 5 == 0) {
            auto input_copy = input.clone();
            auto divisor_val = get_value<float>(Data, Size, offset);
            if (divisor_val == 0.0f) {
                divisor_val = 1.0f;
            }
            input_copy.remainder_(divisor_val);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
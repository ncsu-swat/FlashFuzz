#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation and shift amount
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        auto config = tensor_config.value();
        
        // Create input tensor - use integer dtypes for bitwise operations
        torch::Dtype dtype;
        uint8_t dtype_choice = consume_byte(Data, Size, offset);
        switch (dtype_choice % 6) {
            case 0: dtype = torch::kInt8; break;
            case 1: dtype = torch::kInt16; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kUInt8; break;
            default: dtype = torch::kInt32; break;
        }

        // Create input tensor with integer values
        auto input_tensor = create_tensor(config.shape, dtype, config.device);
        if (!input_tensor.defined()) {
            return 0;
        }

        // Fill tensor with integer data
        fill_tensor_with_data(input_tensor, Data, Size, offset);

        // Test different shift scenarios
        uint8_t test_mode = consume_byte(Data, Size, offset);
        
        switch (test_mode % 4) {
            case 0: {
                // Test with scalar shift amount
                int64_t shift_amount = static_cast<int64_t>(consume_byte(Data, Size, offset)) % 32;
                auto result = torch::bitwise_right_shift(input_tensor, shift_amount);
                break;
            }
            case 1: {
                // Test with tensor shift amount (same shape)
                auto shift_tensor = torch::randint(0, 16, input_tensor.sizes(), 
                                                 torch::TensorOptions().dtype(torch::kInt32).device(config.device));
                auto result = torch::bitwise_right_shift(input_tensor, shift_tensor);
                break;
            }
            case 2: {
                // Test with broadcastable shift tensor
                std::vector<int64_t> shift_shape;
                if (input_tensor.dim() > 0) {
                    // Create a shape that can broadcast
                    for (int i = 0; i < input_tensor.dim(); ++i) {
                        if (i == input_tensor.dim() - 1) {
                            shift_shape.push_back(1); // Make last dimension 1 for broadcasting
                        } else {
                            shift_shape.push_back(input_tensor.size(i));
                        }
                    }
                    auto shift_tensor = torch::randint(0, 8, shift_shape, 
                                                     torch::TensorOptions().dtype(torch::kInt32).device(config.device));
                    auto result = torch::bitwise_right_shift(input_tensor, shift_tensor);
                }
                break;
            }
            case 3: {
                // Test in-place operation
                auto input_copy = input_tensor.clone();
                int64_t shift_amount = static_cast<int64_t>(consume_byte(Data, Size, offset)) % 16;
                input_copy.bitwise_right_shift_(shift_amount);
                break;
            }
        }

        // Test edge cases with specific values
        if (offset < Size - 4) {
            // Test with zero shift
            auto zero_shift_result = torch::bitwise_right_shift(input_tensor, 0);
            
            // Test with maximum reasonable shift for the dtype
            int max_shift = 0;
            if (dtype == torch::kInt8 || dtype == torch::kUInt8) {
                max_shift = 7;
            } else if (dtype == torch::kInt16) {
                max_shift = 15;
            } else if (dtype == torch::kInt32) {
                max_shift = 31;
            } else if (dtype == torch::kInt64) {
                max_shift = 63;
            }
            
            if (max_shift > 0) {
                auto max_shift_result = torch::bitwise_right_shift(input_tensor, max_shift);
            }
        }

        // Test with different tensor shapes if we have enough data
        if (offset < Size - 8 && input_tensor.numel() > 1) {
            // Test with reshaped tensor
            auto total_elements = input_tensor.numel();
            if (total_elements >= 4) {
                try {
                    auto reshaped = input_tensor.view({-1});
                    int64_t shift_val = static_cast<int64_t>(consume_byte(Data, Size, offset)) % 8;
                    auto result = torch::bitwise_right_shift(reshaped, shift_val);
                } catch (...) {
                    // Ignore reshape failures
                }
            }
        }

        // Test with negative shift values (should be handled gracefully or throw)
        if (offset < Size - 2) {
            try {
                int64_t negative_shift = -static_cast<int64_t>(consume_byte(Data, Size, offset) % 8 + 1);
                auto result = torch::bitwise_right_shift(input_tensor, negative_shift);
            } catch (...) {
                // Expected for negative shifts
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
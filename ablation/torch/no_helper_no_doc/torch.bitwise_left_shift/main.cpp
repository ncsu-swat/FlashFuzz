#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and shift amount
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        auto config = tensor_config.value();
        
        // Create input tensor with integer dtype (bitwise operations require integer types)
        torch::ScalarType dtype = torch::kInt32;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++];
            switch (dtype_choice % 6) {
                case 0: dtype = torch::kInt8; break;
                case 1: dtype = torch::kInt16; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kUInt8; break;
                case 5: dtype = torch::kBool; break;
            }
        }

        auto tensor = create_tensor(config, dtype);
        if (!tensor.has_value()) {
            return 0;
        }

        auto input_tensor = tensor.value();

        // Extract shift amount - can be scalar or tensor
        bool use_tensor_shift = false;
        if (offset < Size) {
            use_tensor_shift = (Data[offset++] % 2) == 1;
        }

        if (use_tensor_shift) {
            // Create shift tensor with same shape or broadcastable shape
            auto shift_config = extract_tensor_config(Data, Size, offset);
            if (!shift_config.has_value()) {
                return 0;
            }

            // Use integer dtype for shift tensor
            torch::ScalarType shift_dtype = torch::kInt32;
            if (offset < Size) {
                uint8_t shift_dtype_choice = Data[offset++];
                switch (shift_dtype_choice % 4) {
                    case 0: shift_dtype = torch::kInt8; break;
                    case 1: shift_dtype = torch::kInt16; break;
                    case 2: shift_dtype = torch::kInt32; break;
                    case 3: shift_dtype = torch::kInt64; break;
                }
            }

            auto shift_tensor_opt = create_tensor(shift_config.value(), shift_dtype);
            if (!shift_tensor_opt.has_value()) {
                return 0;
            }

            auto shift_tensor = shift_tensor_opt.value();

            // Test tensor-tensor bitwise left shift
            auto result1 = torch::bitwise_left_shift(input_tensor, shift_tensor);
            
            // Test in-place version
            auto input_copy = input_tensor.clone();
            input_copy.bitwise_left_shift_(shift_tensor);

            // Test with different broadcasting scenarios
            if (input_tensor.dim() > 0 && shift_tensor.dim() > 0) {
                try {
                    // Try broadcasting with reshaped tensors
                    auto reshaped_shift = shift_tensor.view({-1});
                    if (reshaped_shift.numel() == 1) {
                        auto broadcast_result = torch::bitwise_left_shift(input_tensor, reshaped_shift);
                    }
                } catch (...) {
                    // Broadcasting might fail, which is expected
                }
            }

        } else {
            // Use scalar shift amount
            int64_t shift_amount = 0;
            if (offset + sizeof(int32_t) <= Size) {
                shift_amount = *reinterpret_cast<const int32_t*>(Data + offset);
                offset += sizeof(int32_t);
                // Clamp shift amount to reasonable range to avoid undefined behavior
                shift_amount = std::max(-64L, std::min(64L, shift_amount));
            }

            // Test tensor-scalar bitwise left shift
            auto result2 = torch::bitwise_left_shift(input_tensor, shift_amount);
            
            // Test in-place version with scalar
            auto input_copy2 = input_tensor.clone();
            input_copy2.bitwise_left_shift_(shift_amount);

            // Test with different scalar types
            if (offset < Size) {
                uint8_t scalar_type = Data[offset++];
                switch (scalar_type % 4) {
                    case 0: {
                        auto result_int8 = torch::bitwise_left_shift(input_tensor, static_cast<int8_t>(shift_amount));
                        break;
                    }
                    case 1: {
                        auto result_int16 = torch::bitwise_left_shift(input_tensor, static_cast<int16_t>(shift_amount));
                        break;
                    }
                    case 2: {
                        auto result_int32 = torch::bitwise_left_shift(input_tensor, static_cast<int32_t>(shift_amount));
                        break;
                    }
                    case 3: {
                        auto result_int64 = torch::bitwise_left_shift(input_tensor, shift_amount);
                        break;
                    }
                }
            }
        }

        // Test edge cases with special values
        if (input_tensor.numel() > 0) {
            // Test with zero tensor
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto zero_result = torch::bitwise_left_shift(zero_tensor, 1);

            // Test with ones tensor
            auto ones_tensor = torch::ones_like(input_tensor);
            auto ones_result = torch::bitwise_left_shift(ones_tensor, 1);

            // Test shift by zero
            auto no_shift_result = torch::bitwise_left_shift(input_tensor, 0);
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            bool use_cuda = (Data[offset++] % 2) == 1;
            if (use_cuda) {
                try {
                    auto cuda_tensor = input_tensor.to(torch::kCUDA);
                    auto cuda_result = torch::bitwise_left_shift(cuda_tensor, 1);
                } catch (...) {
                    // CUDA operations might fail, which is acceptable
                }
            }
        }

        // Test output tensor variant if there's remaining data
        if (offset < Size) {
            bool test_out_variant = (Data[offset++] % 2) == 1;
            if (test_out_variant) {
                auto out_tensor = torch::empty_like(input_tensor);
                torch::bitwise_left_shift_out(out_tensor, input_tensor, 1);
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
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic operations
        if (Size < 16) {
            return 0;
        }

        // Extract operation mode (0: tensor^scalar, 1: tensor^tensor, 2: scalar^tensor)
        uint8_t mode = consume_uint8_t(Data, Size, offset) % 3;

        if (mode == 0) {
            // Test tensor^scalar case
            auto input_tensor = consume_tensor(Data, Size, offset);
            if (input_tensor.numel() == 0) {
                return 0;
            }

            // Get scalar exponent
            double exponent = consume_float_in_range(Data, Size, offset, -10.0, 10.0);
            
            // Test torch::pow with tensor and scalar
            auto result = torch::pow(input_tensor, exponent);
            
            // Test with different scalar types
            float exponent_f = static_cast<float>(exponent);
            auto result_f = torch::pow(input_tensor, exponent_f);
            
            int exponent_i = static_cast<int>(exponent);
            auto result_i = torch::pow(input_tensor, exponent_i);
            
            // Test edge cases with special values
            if (offset + 1 < Size) {
                uint8_t special_case = consume_uint8_t(Data, Size, offset) % 6;
                double special_exp;
                switch (special_case) {
                    case 0: special_exp = 0.0; break;
                    case 1: special_exp = 1.0; break;
                    case 2: special_exp = -1.0; break;
                    case 3: special_exp = 0.5; break;
                    case 4: special_exp = 2.0; break;
                    case 5: special_exp = std::numeric_limits<double>::infinity(); break;
                }
                auto special_result = torch::pow(input_tensor, special_exp);
            }
        }
        else if (mode == 1) {
            // Test tensor^tensor case
            auto input_tensor = consume_tensor(Data, Size, offset);
            if (input_tensor.numel() == 0) {
                return 0;
            }

            auto exponent_tensor = consume_tensor(Data, Size, offset);
            if (exponent_tensor.numel() == 0) {
                return 0;
            }

            try {
                // Test torch::pow with two tensors (broadcasting)
                auto result = torch::pow(input_tensor, exponent_tensor);
                
                // Test with same shape tensors
                if (input_tensor.sizes() == exponent_tensor.sizes()) {
                    auto same_shape_result = torch::pow(input_tensor, exponent_tensor);
                }
                
                // Test with broadcastable shapes
                auto reshaped_input = input_tensor.view({-1});
                if (reshaped_input.numel() > 0) {
                    auto scalar_exp = exponent_tensor.view({}).item<double>();
                    auto broadcast_result = torch::pow(reshaped_input, scalar_exp);
                }
            } catch (const std::exception& e) {
                // Broadcasting might fail, which is expected behavior
            }
        }
        else {
            // Test scalar^tensor case
            double base = consume_float_in_range(Data, Size, offset, -10.0, 10.0);
            auto exponent_tensor = consume_tensor(Data, Size, offset);
            if (exponent_tensor.numel() == 0) {
                return 0;
            }

            // Test torch::pow with scalar base and tensor exponent
            auto result = torch::pow(base, exponent_tensor);
            
            // Test with different scalar types
            float base_f = static_cast<float>(base);
            auto result_f = torch::pow(base_f, exponent_tensor);
            
            int base_i = static_cast<int>(base);
            auto result_i = torch::pow(base_i, exponent_tensor);
            
            // Test edge cases with special base values
            if (offset + 1 < Size) {
                uint8_t special_case = consume_uint8_t(Data, Size, offset) % 6;
                double special_base;
                switch (special_case) {
                    case 0: special_base = 0.0; break;
                    case 1: special_base = 1.0; break;
                    case 2: special_base = -1.0; break;
                    case 3: special_base = 2.0; break;
                    case 4: special_base = 10.0; break;
                    case 5: special_base = std::numeric_limits<double>::infinity(); break;
                }
                auto special_result = torch::pow(special_base, exponent_tensor);
            }
        }

        // Test with output tensor parameter if we have enough data
        if (offset + 8 < Size) {
            auto input_tensor = consume_tensor(Data, Size, offset);
            if (input_tensor.numel() > 0) {
                double exponent = consume_float_in_range(Data, Size, offset, -5.0, 5.0);
                
                // Create output tensor with same shape as input
                auto out_tensor = torch::empty_like(input_tensor);
                torch::pow_out(out_tensor, input_tensor, exponent);
                
                // Test with different output tensor dtype
                if (input_tensor.dtype() != torch::kFloat64) {
                    auto out_tensor_f64 = torch::empty_like(input_tensor, torch::kFloat64);
                    torch::pow_out(out_tensor_f64, input_tensor.to(torch::kFloat64), exponent);
                }
            }
        }

        // Test in-place operations if we have enough data
        if (offset + 4 < Size) {
            auto input_tensor = consume_tensor(Data, Size, offset);
            if (input_tensor.numel() > 0 && input_tensor.is_floating_point()) {
                double exponent = consume_float_in_range(Data, Size, offset, -3.0, 3.0);
                
                // Make a copy for in-place operation
                auto inplace_tensor = input_tensor.clone();
                inplace_tensor.pow_(exponent);
                
                // Test in-place with tensor exponent
                auto exp_tensor = torch::full_like(input_tensor, exponent);
                auto inplace_tensor2 = input_tensor.clone();
                inplace_tensor2.pow_(exp_tensor);
            }
        }

        // Test with different tensor dtypes and devices
        if (offset + 2 < Size) {
            uint8_t dtype_idx = consume_uint8_t(Data, Size, offset) % 6;
            torch::ScalarType dtype;
            switch (dtype_idx) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kBool; break;
                case 5: dtype = torch::kHalf; break;
            }
            
            auto input_tensor = consume_tensor(Data, Size, offset, dtype);
            if (input_tensor.numel() > 0) {
                double exponent = consume_float_in_range(Data, Size, offset, -2.0, 2.0);
                
                try {
                    auto result = torch::pow(input_tensor, exponent);
                } catch (const std::exception& e) {
                    // Some dtype combinations might not be supported
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
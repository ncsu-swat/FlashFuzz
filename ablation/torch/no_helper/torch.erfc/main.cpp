#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);
        auto shape = extract_shape(Data, Size, offset);
        
        // Skip if shape is too large to avoid memory issues
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
            if (total_elements > 10000) {
                return 0;
            }
        }

        // Create input tensor with various data patterns
        torch::Tensor input;
        
        // Try different tensor creation strategies based on remaining data
        if (offset < Size) {
            uint8_t creation_mode = Data[offset++] % 6;
            
            switch (creation_mode) {
                case 0:
                    // Random tensor
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 1:
                    // Zeros tensor
                    input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 2:
                    // Ones tensor
                    input = torch::ones(shape, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 3:
                    // Large positive values
                    input = torch::full(shape, 10.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 4:
                    // Large negative values
                    input = torch::full(shape, -10.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 5:
                    // Small values around zero
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 0.1;
                    break;
            }
        } else {
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Add some special values if there's remaining data
        if (offset < Size && input.numel() > 0) {
            auto flat_input = input.flatten();
            int64_t num_special = std::min(static_cast<int64_t>(Size - offset), flat_input.numel());
            
            for (int64_t i = 0; i < num_special && offset < Size; ++i, ++offset) {
                uint8_t special_type = Data[offset] % 8;
                double special_val;
                
                switch (special_type) {
                    case 0: special_val = std::numeric_limits<double>::infinity(); break;
                    case 1: special_val = -std::numeric_limits<double>::infinity(); break;
                    case 2: special_val = std::numeric_limits<double>::quiet_NaN(); break;
                    case 3: special_val = 0.0; break;
                    case 4: special_val = -0.0; break;
                    case 5: special_val = std::numeric_limits<double>::max(); break;
                    case 6: special_val = std::numeric_limits<double>::lowest(); break;
                    case 7: special_val = std::numeric_limits<double>::min(); break;
                }
                
                if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
                    flat_input[i] = special_val;
                }
            }
        }

        // Test torch.erfc with basic call
        torch::Tensor result1 = torch::erfc(input);
        
        // Test with output tensor if there's remaining data
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            torch::Tensor out = torch::empty_like(result1);
            torch::erfc_out(out, input);
            
            // Verify output tensor was modified
            if (out.numel() > 0) {
                auto out_sum = out.sum();
                (void)out_sum; // Use the result to prevent optimization
            }
        }

        // Test with different input modifications
        if (offset < Size) {
            uint8_t mod_type = Data[offset++] % 4;
            torch::Tensor modified_input;
            
            switch (mod_type) {
                case 0:
                    // Test with transposed input
                    if (input.dim() >= 2) {
                        modified_input = input.transpose(0, 1);
                        torch::Tensor result2 = torch::erfc(modified_input);
                    }
                    break;
                case 1:
                    // Test with contiguous input
                    modified_input = input.contiguous();
                    torch::Tensor result3 = torch::erfc(modified_input);
                    break;
                case 2:
                    // Test with squeezed input
                    modified_input = input.squeeze();
                    torch::Tensor result4 = torch::erfc(modified_input);
                    break;
                case 3:
                    // Test with unsqueezed input
                    modified_input = input.unsqueeze(0);
                    torch::Tensor result5 = torch::erfc(modified_input);
                    break;
            }
        }

        // Verify result properties
        if (result1.numel() > 0) {
            // Check that result has same shape as input
            if (!result1.sizes().equals(input.sizes())) {
                std::cerr << "Shape mismatch in erfc result" << std::endl;
            }
            
            // Access some elements to trigger computation
            auto result_sum = result1.sum();
            auto result_mean = result1.mean();
            (void)result_sum;
            (void)result_mean;
            
            // Test with scalar input if possible
            if (input.numel() == 1) {
                auto scalar_input = input.item<double>();
                auto scalar_result = torch::erfc(torch::tensor(scalar_input));
                (void)scalar_result;
            }
        }

        // Test edge cases with specific dtypes
        if (offset < Size) {
            uint8_t edge_test = Data[offset++] % 3;
            
            switch (edge_test) {
                case 0:
                    // Test with float32
                    if (input.dtype() != torch::kFloat32) {
                        auto float_input = input.to(torch::kFloat32);
                        auto float_result = torch::erfc(float_input);
                        (void)float_result;
                    }
                    break;
                case 1:
                    // Test with float64
                    if (input.dtype() != torch::kFloat64) {
                        auto double_input = input.to(torch::kFloat64);
                        auto double_result = torch::erfc(double_input);
                        (void)double_result;
                    }
                    break;
                case 2:
                    // Test with cloned input
                    auto cloned_input = input.clone();
                    auto cloned_result = torch::erfc(cloned_input);
                    (void)cloned_result;
                    break;
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
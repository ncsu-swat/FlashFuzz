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
                    input = torch::full(shape, 100.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 4:
                    // Large negative values (i0 should handle negative inputs)
                    input = torch::full(shape, -100.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 5:
                    // Small values near zero
                    input = torch::full(shape, 1e-6, torch::TensorOptions().dtype(dtype).device(device));
                    break;
            }
        } else {
            input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Test torch::i0 with various edge cases
        
        // Basic i0 call
        auto result1 = torch::i0(input);
        
        // Test with output tensor if we have more data
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            auto out_tensor = torch::empty_like(input);
            auto result2 = torch::i0_out(out_tensor, input);
        }
        
        // Test with specific problematic values if tensor is small enough
        if (input.numel() <= 100) {
            // Test with infinity
            if (torch::is_floating_point(input)) {
                auto inf_input = torch::full_like(input, std::numeric_limits<double>::infinity());
                auto inf_result = torch::i0(inf_input);
                
                // Test with negative infinity
                auto neg_inf_input = torch::full_like(input, -std::numeric_limits<double>::infinity());
                auto neg_inf_result = torch::i0(neg_inf_input);
                
                // Test with NaN
                auto nan_input = torch::full_like(input, std::numeric_limits<double>::quiet_NaN());
                auto nan_result = torch::i0(nan_input);
            }
        }
        
        // Test with different input ranges based on remaining data
        if (offset < Size) {
            uint8_t range_mode = Data[offset++] % 4;
            torch::Tensor range_input;
            
            switch (range_mode) {
                case 0:
                    // Very large values
                    range_input = torch::full(shape, 700.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 1:
                    // Very small values
                    range_input = torch::full(shape, 1e-10, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 2:
                    // Negative values
                    range_input = torch::full(shape, -50.0, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 3:
                    // Mixed positive and negative
                    range_input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device)) * 100;
                    break;
            }
            
            auto range_result = torch::i0(range_input);
        }
        
        // Test with scalar tensor
        auto scalar_input = torch::tensor(42.0, torch::TensorOptions().dtype(dtype).device(device));
        auto scalar_result = torch::i0(scalar_input);
        
        // Test with empty tensor if supported
        if (shape.size() > 0) {
            auto empty_shape = shape;
            empty_shape[0] = 0;
            auto empty_input = torch::empty(empty_shape, torch::TensorOptions().dtype(dtype).device(device));
            auto empty_result = torch::i0(empty_input);
        }

        // Force evaluation by accessing a value if tensor is small
        if (result1.numel() > 0 && result1.numel() <= 10) {
            auto item = result1.flatten()[0].item<double>();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
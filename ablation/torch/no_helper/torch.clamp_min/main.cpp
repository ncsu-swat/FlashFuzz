#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and min value
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (!tensor_info.has_value()) {
            return 0;
        }

        // Create input tensor with various data types and shapes
        torch::Tensor input_tensor;
        try {
            input_tensor = create_tensor_from_data(Data, Size, offset, tensor_info.value());
        } catch (...) {
            return 0;
        }

        // Extract min value - try different approaches
        if (offset >= Size) {
            return 0;
        }

        // Test 1: clamp_min with scalar value
        if (offset + sizeof(float) <= Size) {
            float min_val = extract_float(Data, offset);
            offset += sizeof(float);
            
            // Handle special float values
            if (std::isnan(min_val)) {
                min_val = 0.0f; // Replace NaN with valid value
            }
            if (std::isinf(min_val)) {
                min_val = std::signbit(min_val) ? -1000.0f : 1000.0f;
            }

            auto result1 = torch::clamp_min(input_tensor, min_val);
            
            // Test in-place version
            auto input_copy = input_tensor.clone();
            torch::clamp_min_(input_copy, min_val);
        }

        // Test 2: clamp_min with tensor min value
        if (offset + 4 <= Size) {
            // Create a min tensor with compatible shape
            std::vector<int64_t> min_shape;
            uint8_t min_dims = Data[offset] % 4; // 0-3 dimensions
            offset++;
            
            for (int i = 0; i < min_dims && offset < Size; i++) {
                int64_t dim_size = (Data[offset] % 10) + 1; // 1-10
                min_shape.push_back(dim_size);
                offset++;
            }

            if (min_shape.empty()) {
                min_shape.push_back(1); // At least one element
            }

            try {
                torch::Tensor min_tensor;
                if (offset + sizeof(float) <= Size) {
                    float min_val = extract_float(Data, offset);
                    offset += sizeof(float);
                    
                    if (std::isnan(min_val) || std::isinf(min_val)) {
                        min_val = 0.0f;
                    }
                    
                    min_tensor = torch::full(min_shape, min_val, input_tensor.options());
                } else {
                    min_tensor = torch::zeros(min_shape, input_tensor.options());
                }

                auto result2 = torch::clamp_min(input_tensor, min_tensor);
                
                // Test in-place version with tensor
                auto input_copy2 = input_tensor.clone();
                torch::clamp_min_(input_copy2, min_tensor);
            } catch (...) {
                // Skip if tensor creation fails
            }
        }

        // Test 3: Edge cases with different tensor properties
        if (input_tensor.numel() > 0) {
            // Test with very small and large values
            auto result_small = torch::clamp_min(input_tensor, -1e6);
            auto result_large = torch::clamp_min(input_tensor, 1e6);
            
            // Test with zero
            auto result_zero = torch::clamp_min(input_tensor, 0.0);
            
            // Test with negative values
            auto result_neg = torch::clamp_min(input_tensor, -100.0);
        }

        // Test 4: Different tensor dtypes if possible
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset] % 6;
            offset++;
            
            torch::ScalarType target_dtype;
            switch (dtype_choice) {
                case 0: target_dtype = torch::kFloat32; break;
                case 1: target_dtype = torch::kFloat64; break;
                case 2: target_dtype = torch::kInt32; break;
                case 3: target_dtype = torch::kInt64; break;
                case 4: target_dtype = torch::kInt16; break;
                default: target_dtype = torch::kInt8; break;
            }
            
            try {
                auto typed_tensor = input_tensor.to(target_dtype);
                auto result_typed = torch::clamp_min(typed_tensor, 0);
            } catch (...) {
                // Skip if conversion fails
            }
        }

        // Test 5: Broadcasting scenarios
        if (input_tensor.dim() > 1 && offset < Size) {
            try {
                // Create a min tensor that can broadcast
                auto min_broadcast = torch::ones({1}, input_tensor.options()) * (Data[offset] % 100 - 50);
                offset++;
                
                auto result_broadcast = torch::clamp_min(input_tensor, min_broadcast);
            } catch (...) {
                // Skip if broadcasting fails
            }
        }

        // Test 6: Empty tensor edge case
        try {
            auto empty_tensor = torch::empty({0}, input_tensor.options());
            auto result_empty = torch::clamp_min(empty_tensor, 1.0);
        } catch (...) {
            // Skip if empty tensor operations fail
        }

        // Test 7: Single element tensor
        try {
            auto single_tensor = torch::tensor({42.0}, input_tensor.options());
            auto result_single = torch::clamp_min(single_tensor, 50.0);
            auto result_single2 = torch::clamp_min(single_tensor, 30.0);
        } catch (...) {
            // Skip if single element operations fail
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
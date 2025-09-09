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

        // Extract tensor dimensions and properties
        auto shape_info = extract_tensor_shape(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (shape_info.empty()) {
            return 0;
        }

        // Extract dtype
        auto dtype = extract_dtype(Data, Size, offset);
        
        // Extract device type
        auto device = extract_device(Data, Size, offset);

        // Create input tensor with the extracted properties
        auto input_tensor = create_tensor(shape_info, dtype, device, Data, Size, offset);
        if (!input_tensor.defined()) {
            return 0;
        }

        // Test torch::trace with different scenarios
        
        // Case 1: Basic trace operation on 2D matrix
        if (input_tensor.dim() >= 2) {
            auto trace_result = torch::trace(input_tensor);
            
            // Verify result is a scalar
            if (trace_result.dim() != 0) {
                std::cerr << "Trace result should be scalar but has " << trace_result.dim() << " dimensions" << std::endl;
            }
        }

        // Case 2: Test with reshaped tensor to ensure it's 2D
        if (input_tensor.numel() >= 4) {
            // Reshape to a square matrix if possible
            int64_t sqrt_numel = static_cast<int64_t>(std::sqrt(input_tensor.numel()));
            if (sqrt_numel * sqrt_numel <= input_tensor.numel()) {
                auto reshaped = input_tensor.view({sqrt_numel, sqrt_numel});
                auto trace_result = torch::trace(reshaped);
                
                // Verify the trace is computed correctly for square matrix
                if (trace_result.dim() != 0) {
                    std::cerr << "Trace of square matrix should be scalar" << std::endl;
                }
            }
        }

        // Case 3: Test with rectangular matrix
        if (input_tensor.numel() >= 6) {
            auto reshaped = input_tensor.view({2, -1});
            if (reshaped.size(1) >= 2) {
                auto trace_result = torch::trace(reshaped);
                // For rectangular matrix, trace should still work
            }
        }

        // Case 4: Test with different data types if tensor supports it
        if (input_tensor.dtype() != torch::kFloat32) {
            try {
                auto float_tensor = input_tensor.to(torch::kFloat32);
                if (float_tensor.dim() >= 2) {
                    auto trace_result = torch::trace(float_tensor);
                }
            } catch (...) {
                // Conversion might fail for some types, that's okay
            }
        }

        // Case 5: Test with transposed matrix
        if (input_tensor.dim() == 2) {
            auto transposed = input_tensor.transpose(0, 1);
            auto trace_result = torch::trace(transposed);
        }

        // Case 6: Test with contiguous and non-contiguous tensors
        if (input_tensor.dim() >= 2) {
            // Make non-contiguous if possible
            if (input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
                auto sliced = input_tensor.slice(0, 0, input_tensor.size(0), 2);
                if (sliced.dim() >= 2 && sliced.size(0) > 0 && sliced.size(1) > 0) {
                    auto trace_result = torch::trace(sliced);
                }
            }
        }

        // Case 7: Test edge cases with minimum size matrices
        try {
            auto small_tensor = torch::ones({1, 1}, dtype).to(device);
            auto trace_result = torch::trace(small_tensor);
            
            // Verify 1x1 matrix trace equals the single element
            if (!torch::allclose(trace_result, small_tensor.squeeze())) {
                std::cerr << "1x1 matrix trace should equal the element" << std::endl;
            }
        } catch (...) {
            // Device might not support tensor creation
        }

        // Case 8: Test with zero tensor
        if (input_tensor.dim() >= 2) {
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto trace_result = torch::trace(zero_tensor);
            
            // Trace of zero matrix should be zero
            if (!torch::allclose(trace_result, torch::zeros({}, trace_result.dtype()).to(trace_result.device()))) {
                std::cerr << "Trace of zero matrix should be zero" << std::endl;
            }
        }

        // Case 9: Test with identity matrix
        try {
            int64_t min_dim = std::min(input_tensor.size(0), input_tensor.size(1));
            if (min_dim > 0 && min_dim <= 100) { // Reasonable size limit
                auto eye_tensor = torch::eye(min_dim, dtype).to(device);
                auto trace_result = torch::trace(eye_tensor);
                
                // Trace of identity matrix should equal the dimension
                auto expected = torch::tensor(static_cast<double>(min_dim), dtype).to(device);
                if (!torch::allclose(trace_result, expected, 1e-5, 1e-5)) {
                    std::cerr << "Trace of identity matrix should equal dimension" << std::endl;
                }
            }
        } catch (...) {
            // Eye tensor creation might fail
        }

        // Case 10: Test with very large matrices (if memory allows)
        if (input_tensor.numel() > 10000) {
            // Only test if we have enough elements
            int64_t side = static_cast<int64_t>(std::sqrt(input_tensor.numel() / 4));
            if (side > 10 && side < 1000) { // Reasonable bounds
                try {
                    auto large_tensor = input_tensor.view({side, side});
                    auto trace_result = torch::trace(large_tensor);
                } catch (...) {
                    // Memory allocation might fail
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
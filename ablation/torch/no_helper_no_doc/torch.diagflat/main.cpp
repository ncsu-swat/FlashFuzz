#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and offset parameter
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions and properties
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (tensor_info.dims.empty()) {
            return 0;
        }

        // Create input tensor with various shapes and data types
        torch::Tensor input;
        try {
            input = create_tensor(tensor_info);
        } catch (...) {
            return 0;
        }

        // Extract offset parameter for diagflat
        int64_t offset_param = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t raw_offset = *reinterpret_cast<const int32_t*>(Data + offset);
            offset += sizeof(int32_t);
            // Limit offset to reasonable range to avoid excessive memory usage
            offset_param = static_cast<int64_t>(raw_offset) % 1000;
        }

        // Test torch::diagflat with different input configurations
        
        // Test 1: Basic diagflat with offset
        torch::Tensor result1 = torch::diagflat(input, offset_param);
        
        // Test 2: diagflat with zero offset
        torch::Tensor result2 = torch::diagflat(input, 0);
        
        // Test 3: diagflat with negative offset
        torch::Tensor result3 = torch::diagflat(input, -offset_param);
        
        // Test 4: Test with flattened input (1D tensor)
        torch::Tensor flattened = input.flatten();
        torch::Tensor result4 = torch::diagflat(flattened, offset_param);
        
        // Test 5: Test with scalar input
        if (input.numel() > 0) {
            torch::Tensor scalar = input.flatten()[0];
            torch::Tensor result5 = torch::diagflat(scalar, offset_param);
        }
        
        // Test 6: Test with different data types if possible
        if (input.dtype() != torch::kFloat32) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor result6 = torch::diagflat(float_input, offset_param);
            } catch (...) {
                // Ignore conversion errors
            }
        }
        
        // Test 7: Test with complex numbers if supported
        if (input.dtype().isFloatingPoint()) {
            try {
                torch::Tensor complex_input = torch::complex(input, torch::zeros_like(input));
                torch::Tensor result7 = torch::diagflat(complex_input, offset_param);
            } catch (...) {
                // Ignore if complex operations not supported
            }
        }
        
        // Test 8: Edge case with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor result8 = torch::diagflat(empty_tensor, offset_param);
        } catch (...) {
            // Expected to potentially fail
        }
        
        // Test 9: Large offset values (within reasonable bounds)
        if (input.numel() > 0) {
            try {
                int64_t large_offset = std::min(static_cast<int64_t>(input.numel()), 100L);
                torch::Tensor result9 = torch::diagflat(input, large_offset);
                torch::Tensor result10 = torch::diagflat(input, -large_offset);
            } catch (...) {
                // May fail due to memory constraints
            }
        }
        
        // Test 10: Verify basic properties of results
        if (result1.defined()) {
            // Result should be 2D
            if (result1.dim() != 2) {
                std::cout << "Unexpected result dimension: " << result1.dim() << std::endl;
            }
            
            // Result should be square matrix
            if (result1.size(0) != result1.size(1)) {
                std::cout << "Result is not square matrix" << std::endl;
            }
            
            // Access some elements to trigger potential issues
            if (result1.numel() > 0) {
                auto accessor = result1.accessor<float, 2>();
                volatile auto val = accessor[0][0];
                (void)val; // Suppress unused variable warning
            }
        }
        
        // Test 11: Test with different tensor layouts/strides
        if (input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(-1, -2);
                torch::Tensor result11 = torch::diagflat(transposed, offset_param);
            } catch (...) {
                // May fail for certain configurations
            }
        }
        
        // Test 12: Test with non-contiguous tensors
        if (input.numel() > 1) {
            try {
                torch::Tensor sliced = input.flatten()[torch::indexing::Slice(0, torch::indexing::None, 2)];
                torch::Tensor result12 = torch::diagflat(sliced, offset_param);
            } catch (...) {
                // May fail for certain configurations
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
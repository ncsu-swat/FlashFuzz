#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and data types
        auto input_tensor = generateTensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Test 1: Basic std without any parameters
        auto result1 = torch::std(input_tensor);

        // Test 2: std with correction parameter
        if (offset < Size) {
            int correction = static_cast<int>(Data[offset % Size]) % 5; // 0-4 range
            offset++;
            auto result2 = torch::std(input_tensor, /*dim=*/c10::nullopt, correction);
        }

        // Test 3: std with keepdim parameter
        if (offset < Size) {
            bool keepdim = (Data[offset % Size] % 2) == 0;
            offset++;
            auto result3 = torch::std(input_tensor, /*dim=*/c10::nullopt, /*correction=*/1, keepdim);
        }

        // Test 4: std with single dimension
        if (input_tensor.dim() > 0 && offset < Size) {
            int dim = static_cast<int>(Data[offset % Size]) % input_tensor.dim();
            offset++;
            auto result4 = torch::std(input_tensor, dim);
            
            // Test with keepdim=true for single dimension
            if (offset < Size) {
                bool keepdim = (Data[offset % Size] % 2) == 0;
                offset++;
                auto result5 = torch::std(input_tensor, dim, /*correction=*/1, keepdim);
            }
        }

        // Test 5: std with multiple dimensions
        if (input_tensor.dim() > 1 && offset + 1 < Size) {
            std::vector<int64_t> dims;
            int num_dims = (Data[offset % Size] % input_tensor.dim()) + 1;
            offset++;
            
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int dim = static_cast<int>(Data[offset % Size]) % input_tensor.dim();
                // Avoid duplicate dimensions
                if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                    dims.push_back(dim);
                }
                offset++;
            }
            
            if (!dims.empty()) {
                auto result6 = torch::std(input_tensor, dims);
                
                // Test with correction and keepdim
                if (offset + 1 < Size) {
                    int correction = static_cast<int>(Data[offset % Size]) % 5;
                    offset++;
                    bool keepdim = (Data[offset % Size] % 2) == 0;
                    offset++;
                    auto result7 = torch::std(input_tensor, dims, correction, keepdim);
                }
            }
        }

        // Test 6: std with negative dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            int dim = -(static_cast<int>(Data[offset % Size]) % input_tensor.dim()) - 1;
            offset++;
            auto result8 = torch::std(input_tensor, dim);
        }

        // Test 7: Edge case with correction = 0 (population std)
        if (offset < Size) {
            auto result9 = torch::std(input_tensor, /*dim=*/c10::nullopt, /*correction=*/0);
        }

        // Test 8: Large correction value
        if (offset < Size) {
            int large_correction = static_cast<int>(Data[offset % Size]) % 100 + 10;
            offset++;
            auto result10 = torch::std(input_tensor, /*dim=*/c10::nullopt, large_correction);
        }

        // Test 9: std with output tensor (if we can create a compatible one)
        if (offset < Size) {
            try {
                auto output_shape = input_tensor.sizes().vec();
                auto out_tensor = torch::empty(output_shape, input_tensor.options());
                torch::std_out(out_tensor, input_tensor);
            } catch (...) {
                // Output tensor might not be compatible, ignore
            }
        }

        // Test 10: Test with different tensor types if possible
        if (offset < Size && input_tensor.dtype() != torch::kFloat32) {
            try {
                auto float_tensor = input_tensor.to(torch::kFloat32);
                auto result11 = torch::std(float_tensor);
            } catch (...) {
                // Type conversion might fail, ignore
            }
        }

        // Test 11: Test with very small tensors
        if (input_tensor.numel() == 1) {
            auto result12 = torch::std(input_tensor);
        }

        // Test 12: Test all dimensions at once
        if (input_tensor.dim() > 1) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                all_dims.push_back(i);
            }
            auto result13 = torch::std(input_tensor, all_dims);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
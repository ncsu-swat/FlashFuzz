#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor dimensions and data
        auto input_dims = generateRandomDimensions(Data, Size, offset, 1, 4);
        if (input_dims.empty()) return 0;
        
        auto input_tensor = generateRandomTensor(Data, Size, offset, input_dims);
        if (!input_tensor.defined()) return 0;

        // Generate test_elements tensor dimensions and data
        auto test_dims = generateRandomDimensions(Data, Size, offset, 1, 4);
        if (test_dims.empty()) return 0;
        
        auto test_tensor = generateRandomTensor(Data, Size, offset, test_dims);
        if (!test_tensor.defined()) return 0;

        // Test basic isin functionality
        auto result1 = torch::isin(input_tensor, test_tensor);
        
        // Test with assume_unique parameter
        if (offset < Size) {
            bool assume_unique = (Data[offset++] % 2) == 0;
            auto result2 = torch::isin(input_tensor, test_tensor, assume_unique);
        }

        // Test with invert parameter
        if (offset < Size) {
            bool invert = (Data[offset++] % 2) == 0;
            auto result3 = torch::isin(input_tensor, test_tensor, false, invert);
        }

        // Test with both assume_unique and invert parameters
        if (offset + 1 < Size) {
            bool assume_unique = (Data[offset++] % 2) == 0;
            bool invert = (Data[offset++] % 2) == 0;
            auto result4 = torch::isin(input_tensor, test_tensor, assume_unique, invert);
        }

        // Test edge cases with different tensor types
        if (offset < Size) {
            auto dtype_idx = Data[offset++] % 6;
            torch::ScalarType dtype;
            switch (dtype_idx) {
                case 0: dtype = torch::kInt32; break;
                case 1: dtype = torch::kInt64; break;
                case 2: dtype = torch::kFloat32; break;
                case 3: dtype = torch::kFloat64; break;
                case 4: dtype = torch::kBool; break;
                default: dtype = torch::kInt8; break;
            }
            
            auto typed_input = input_tensor.to(dtype);
            auto typed_test = test_tensor.to(dtype);
            auto result5 = torch::isin(typed_input, typed_test);
        }

        // Test with scalar test_elements
        if (offset < Size && input_tensor.numel() > 0) {
            auto scalar_val = input_tensor.flatten()[0];
            auto result6 = torch::isin(input_tensor, scalar_val);
        }

        // Test with empty tensors
        auto empty_input = torch::empty({0});
        auto empty_test = torch::empty({0});
        auto result7 = torch::isin(empty_input, empty_test);
        auto result8 = torch::isin(input_tensor, empty_test);
        auto result9 = torch::isin(empty_input, test_tensor);

        // Test with single element tensors
        auto single_input = torch::tensor({1.0});
        auto single_test = torch::tensor({1.0});
        auto result10 = torch::isin(single_input, single_test);

        // Test with large range of values to stress test assume_unique
        if (offset < Size) {
            auto large_input = torch::randint(0, 1000, {100});
            auto large_test = torch::randint(0, 100, {50});
            bool assume_unique = (Data[offset++] % 2) == 0;
            auto result11 = torch::isin(large_input, large_test, assume_unique);
        }

        // Test with duplicate values when assume_unique is true
        if (test_tensor.numel() > 1) {
            auto dup_test = torch::cat({test_tensor, test_tensor});
            auto result12 = torch::isin(input_tensor, dup_test, true);
        }

        // Test with different device placements if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            bool use_cuda = (Data[offset++] % 2) == 0;
            if (use_cuda) {
                auto cuda_input = input_tensor.to(torch::kCUDA);
                auto cuda_test = test_tensor.to(torch::kCUDA);
                auto result13 = torch::isin(cuda_input, cuda_test);
            }
        }

        // Test with mixed precision
        if (offset < Size) {
            auto float_input = input_tensor.to(torch::kFloat32);
            auto double_test = test_tensor.to(torch::kFloat64);
            auto result14 = torch::isin(float_input, double_test);
        }

        // Test with very large tensors (memory stress test)
        if (offset < Size && (Data[offset++] % 10) == 0) {
            try {
                auto large_dims = std::vector<int64_t>{1000, 100};
                auto very_large_input = torch::randint(0, 10000, large_dims);
                auto very_large_test = torch::randint(0, 1000, {500});
                auto result15 = torch::isin(very_large_input, very_large_test);
            } catch (const std::bad_alloc&) {
                // Skip if not enough memory
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
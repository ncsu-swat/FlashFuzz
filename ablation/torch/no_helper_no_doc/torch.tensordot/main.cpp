#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensordot
        if (Size < 20) return 0;

        // Parse tensor dimensions and shapes
        auto shape1_size = consume_integral_in_range<int>(Data, Size, offset, 1, 6);
        auto shape2_size = consume_integral_in_range<int>(Data, Size, offset, 1, 6);
        
        std::vector<int64_t> shape1, shape2;
        for (int i = 0; i < shape1_size; i++) {
            shape1.push_back(consume_integral_in_range<int64_t>(Data, Size, offset, 1, 10));
        }
        for (int i = 0; i < shape2_size; i++) {
            shape2.push_back(consume_integral_in_range<int64_t>(Data, Size, offset, 1, 10));
        }

        // Create tensors with random data
        auto tensor1 = torch::randn(shape1);
        auto tensor2 = torch::randn(shape2);

        // Test different ways to specify dimensions for contraction
        auto test_mode = consume_integral_in_range<int>(Data, Size, offset, 0, 3);

        if (test_mode == 0) {
            // Test with integer dims parameter (contract last N dimensions)
            auto dims = consume_integral_in_range<int>(Data, Size, offset, 0, 
                std::min(static_cast<int>(shape1.size()), static_cast<int>(shape2.size())));
            auto result = torch::tensordot(tensor1, tensor2, dims);
        }
        else if (test_mode == 1) {
            // Test with explicit dimension lists
            auto dims1_count = consume_integral_in_range<int>(Data, Size, offset, 0, 
                std::min(3, static_cast<int>(shape1.size())));
            auto dims2_count = dims1_count; // Must match for valid contraction
            
            std::vector<int64_t> dims1, dims2;
            for (int i = 0; i < dims1_count; i++) {
                dims1.push_back(consume_integral_in_range<int64_t>(Data, Size, offset, 0, shape1.size() - 1));
                dims2.push_back(consume_integral_in_range<int64_t>(Data, Size, offset, 0, shape2.size() - 1));
            }
            
            auto result = torch::tensordot(tensor1, tensor2, {dims1, dims2});
        }
        else if (test_mode == 2) {
            // Test edge cases with empty tensors
            auto empty1 = torch::empty({0});
            auto empty2 = torch::empty({0});
            try {
                auto result = torch::tensordot(empty1, empty2, 0);
            } catch (...) {
                // Expected to potentially fail
            }
        }
        else {
            // Test with different dtypes
            auto dtype_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 4);
            torch::Dtype dtype;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                default: dtype = torch::kFloat16; break;
            }
            
            auto typed_tensor1 = tensor1.to(dtype);
            auto typed_tensor2 = tensor2.to(dtype);
            auto dims = consume_integral_in_range<int>(Data, Size, offset, 0, 2);
            auto result = torch::tensordot(typed_tensor1, typed_tensor2, dims);
        }

        // Test additional edge cases
        if (offset < Size) {
            // Test with 1D tensors
            auto vec1 = torch::randn({5});
            auto vec2 = torch::randn({5});
            auto dot_result = torch::tensordot(vec1, vec2, 1);
            
            // Test with higher dimensional tensors
            auto tensor3d_1 = torch::randn({2, 3, 4});
            auto tensor3d_2 = torch::randn({4, 5, 6});
            auto result_3d = torch::tensordot(tensor3d_1, tensor3d_2, 1);
            
            // Test with mismatched dimensions (should handle gracefully)
            try {
                auto mismatch_result = torch::tensordot(tensor3d_1, tensor3d_2, 2);
            } catch (...) {
                // Expected to potentially fail with dimension mismatch
            }
        }

        // Test with complex tensors if available
        if (offset < Size) {
            auto complex1 = torch::randn({3, 4}, torch::kComplexFloat);
            auto complex2 = torch::randn({4, 5}, torch::kComplexFloat);
            auto complex_result = torch::tensordot(complex1, complex2, 1);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
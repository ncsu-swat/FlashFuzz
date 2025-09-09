#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate two input tensors with various configurations
        auto input_tensor = generate_tensor(Data, Size, offset);
        auto other_tensor = generate_tensor(Data, Size, offset);
        
        // Test basic maximum operation
        auto result1 = torch::maximum(input_tensor, other_tensor);
        
        // Test with broadcasting - create tensors of different shapes
        if (offset < Size) {
            auto broadcast_tensor1 = generate_tensor(Data, Size, offset, {1, 3});
            auto broadcast_tensor2 = generate_tensor(Data, Size, offset, {2, 1});
            auto result2 = torch::maximum(broadcast_tensor1, broadcast_tensor2);
        }
        
        // Test with scalar-like tensor (single element)
        if (offset < Size) {
            auto scalar_tensor = generate_tensor(Data, Size, offset, {1});
            auto result3 = torch::maximum(input_tensor, scalar_tensor);
            auto result4 = torch::maximum(scalar_tensor, input_tensor);
        }
        
        // Test with different dtypes
        if (offset < Size) {
            auto int_tensor1 = generate_tensor(Data, Size, offset, {}, torch::kInt32);
            auto int_tensor2 = generate_tensor(Data, Size, offset, {}, torch::kInt32);
            auto result5 = torch::maximum(int_tensor1, int_tensor2);
        }
        
        if (offset < Size) {
            auto long_tensor1 = generate_tensor(Data, Size, offset, {}, torch::kInt64);
            auto long_tensor2 = generate_tensor(Data, Size, offset, {}, torch::kInt64);
            auto result6 = torch::maximum(long_tensor1, long_tensor2);
        }
        
        // Test with tensors containing special values (inf, -inf, nan)
        if (offset < Size) {
            auto special_tensor1 = torch::tensor({std::numeric_limits<float>::infinity(), 
                                                -std::numeric_limits<float>::infinity(),
                                                std::numeric_limits<float>::quiet_NaN(),
                                                0.0f, 1.0f});
            auto special_tensor2 = torch::tensor({1.0f, 1.0f, 1.0f, 
                                                std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::infinity()});
            auto result7 = torch::maximum(special_tensor1, special_tensor2);
        }
        
        // Test with empty tensors
        if (offset < Size) {
            auto empty1 = torch::empty({0});
            auto empty2 = torch::empty({0});
            auto result8 = torch::maximum(empty1, empty2);
        }
        
        // Test with large tensors
        if (offset < Size) {
            auto large_tensor1 = generate_tensor(Data, Size, offset, {100, 100});
            auto large_tensor2 = generate_tensor(Data, Size, offset, {100, 100});
            auto result9 = torch::maximum(large_tensor1, large_tensor2);
        }
        
        // Test with multi-dimensional tensors
        if (offset < Size) {
            auto multi_dim1 = generate_tensor(Data, Size, offset, {2, 3, 4});
            auto multi_dim2 = generate_tensor(Data, Size, offset, {2, 3, 4});
            auto result10 = torch::maximum(multi_dim1, multi_dim2);
        }
        
        // Test with tensors on different devices (if CUDA available)
        if (torch::cuda::is_available() && offset < Size) {
            auto cuda_tensor1 = generate_tensor(Data, Size, offset).cuda();
            auto cuda_tensor2 = generate_tensor(Data, Size, offset).cuda();
            auto result11 = torch::maximum(cuda_tensor1, cuda_tensor2);
        }
        
        // Test with output parameter
        if (offset < Size) {
            auto out_tensor = torch::empty_like(input_tensor);
            torch::maximum_out(out_tensor, input_tensor, other_tensor);
        }
        
        // Test edge case: same tensor as both inputs
        auto result12 = torch::maximum(input_tensor, input_tensor);
        
        // Test with boolean tensors
        if (offset < Size) {
            auto bool_tensor1 = generate_tensor(Data, Size, offset, {}, torch::kBool);
            auto bool_tensor2 = generate_tensor(Data, Size, offset, {}, torch::kBool);
            auto result13 = torch::maximum(bool_tensor1, bool_tensor2);
        }
        
        // Test with different memory layouts (contiguous vs non-contiguous)
        if (offset < Size) {
            auto tensor_2d = generate_tensor(Data, Size, offset, {4, 4});
            auto transposed = tensor_2d.transpose(0, 1);
            auto result14 = torch::maximum(tensor_2d, transposed);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
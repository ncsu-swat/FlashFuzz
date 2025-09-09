#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and properties
        auto dims1 = parse_tensor_dims(Data, Size, offset);
        if (dims1.empty()) return 0;
        
        auto dims2 = parse_tensor_dims(Data, Size, offset);
        if (dims2.empty()) return 0;

        // Parse dtype - bitwise_and works with integer and boolean types
        auto dtype = parse_dtype(Data, Size, offset);
        // Ensure we use integer or boolean types for bitwise operations
        if (dtype != torch::kInt8 && dtype != torch::kInt16 && dtype != torch::kInt32 && 
            dtype != torch::kInt64 && dtype != torch::kUInt8 && dtype != torch::kBool) {
            dtype = torch::kInt32; // Default to int32
        }

        // Create tensors with the parsed dimensions and dtype
        torch::Tensor tensor1, tensor2;
        
        if (dtype == torch::kBool) {
            tensor1 = torch::randint(0, 2, dims1, torch::TensorOptions().dtype(dtype));
            tensor2 = torch::randint(0, 2, dims2, torch::TensorOptions().dtype(dtype));
        } else {
            // For integer types, use a reasonable range to avoid overflow issues
            int64_t max_val = 1000;
            tensor1 = torch::randint(-max_val, max_val, dims1, torch::TensorOptions().dtype(dtype));
            tensor2 = torch::randint(-max_val, max_val, dims2, torch::TensorOptions().dtype(dtype));
        }

        // Test basic bitwise_and operation
        auto result1 = torch::bitwise_and(tensor1, tensor2);

        // Test with scalar
        if (offset < Size) {
            int64_t scalar_val = static_cast<int64_t>(Data[offset % Size]);
            auto result2 = torch::bitwise_and(tensor1, scalar_val);
        }

        // Test in-place operation if tensors are broadcastable
        try {
            auto tensor1_copy = tensor1.clone();
            if (tensor1_copy.sizes() == tensor2.sizes() || 
                torch::broadcast_shapes(tensor1_copy.sizes(), tensor2.sizes()).has_value()) {
                tensor1_copy.bitwise_and_(tensor2);
            }
        } catch (...) {
            // Ignore broadcasting errors for in-place operations
        }

        // Test with different tensor shapes to trigger broadcasting
        if (offset + 1 < Size) {
            auto scalar_tensor = torch::tensor(static_cast<int64_t>(Data[offset]), 
                                             torch::TensorOptions().dtype(dtype));
            auto result3 = torch::bitwise_and(tensor1, scalar_tensor);
        }

        // Test edge cases with zero-dimensional tensors
        auto zero_dim = torch::tensor(42, torch::TensorOptions().dtype(dtype));
        auto result4 = torch::bitwise_and(zero_dim, tensor1);

        // Test with empty tensors
        if (!dims1.empty() && dims1[0] > 0) {
            auto empty_dims = dims1;
            empty_dims[0] = 0;
            auto empty_tensor = torch::empty(empty_dims, torch::TensorOptions().dtype(dtype));
            auto result5 = torch::bitwise_and(empty_tensor, empty_tensor);
        }

        // Test with single element tensors
        auto single_elem1 = torch::tensor(1, torch::TensorOptions().dtype(dtype));
        auto single_elem2 = torch::tensor(0, torch::TensorOptions().dtype(dtype));
        auto result6 = torch::bitwise_and(single_elem1, single_elem2);

        // Test chained operations
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            auto chained = torch::bitwise_and(torch::bitwise_and(tensor1, tensor2), tensor1);
        }

        // Test with extreme values for integer types
        if (dtype != torch::kBool && offset + 4 < Size) {
            auto extreme_tensor = torch::full_like(tensor1, std::numeric_limits<int32_t>::max());
            auto result7 = torch::bitwise_and(tensor1, extreme_tensor);
            
            auto min_tensor = torch::full_like(tensor1, std::numeric_limits<int32_t>::min());
            auto result8 = torch::bitwise_and(tensor1, min_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
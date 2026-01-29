#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use the same tensor for multiplication
            tensor2 = tensor1;
        }
        
        // Try scalar multiplication if we have more data
        if (offset < Size) {
            // Use next byte as a scalar value
            double scalar_value = static_cast<double>(Data[offset++]);
            
            // Test tensor * scalar
            try {
                torch::Tensor result1 = torch::multiply(tensor1, scalar_value);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Try tensor-tensor multiplication
        // This will test broadcasting rules and handle different shapes
        try {
            torch::Tensor result3 = torch::multiply(tensor1, tensor2);
        } catch (...) {
            // Silently ignore broadcasting failures
        }
        
        // Try in-place multiplication if we have more data
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Clone to avoid modifying the original tensor
                torch::Tensor tensor_copy = tensor1.clone();
                tensor_copy.mul_(tensor2);
            } catch (...) {
                // Silently ignore in-place operation failures
            }
        }
        
        // Try different variants of the multiply API
        try {
            torch::Tensor result4 = tensor1 * tensor2;
        } catch (...) {
            // Silently ignore
        }
        
        try {
            torch::Tensor result5 = torch::mul(tensor1, tensor2);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with empty tensors
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor empty_tensor = torch::empty({0});
                torch::Tensor result_empty = torch::multiply(empty_tensor, empty_tensor);
            } catch (...) {
                // Silently ignore empty tensor issues
            }
        }
        
        // Test with tensors of different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            try {
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                torch::Tensor converted = tensor1.to(dtype);
                torch::Tensor result_mixed_types = torch::multiply(converted, tensor2);
            } catch (...) {
                // Silently ignore dtype conversion/multiplication failures
            }
        }
        
        // Test with output tensor (out parameter variant)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                torch::Tensor out = torch::empty_like(tensor1);
                torch::multiply_out(out, tensor1, tensor1);
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
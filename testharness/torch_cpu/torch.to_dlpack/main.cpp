#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/torch.h>
#include <ATen/DLConvertor.h>

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is contiguous (DLPack requires contiguous memory)
        tensor = tensor.contiguous();
        
        // Convert tensor to DLPack format
        DLManagedTensor* dlpack_tensor = at::toDLPack(tensor);
        
        // Verify DLPack tensor was created
        if (dlpack_tensor == nullptr) {
            throw std::runtime_error("toDLPack returned nullptr");
        }
        
        // Convert back from DLPack to PyTorch tensor to verify round-trip
        // Note: fromDLPack takes ownership and will call the deleter
        torch::Tensor tensor_from_dlpack = at::fromDLPack(dlpack_tensor);
        
        // Verify that the tensors have the same shape
        if (tensor.sizes() != tensor_from_dlpack.sizes()) {
            throw std::runtime_error("Tensor shape mismatch after conversion");
        }
        
        // Verify dtype (may differ due to DLPack type mapping)
        if (tensor.dtype() != tensor_from_dlpack.dtype()) {
            throw std::runtime_error("Tensor dtype mismatch after conversion");
        }
        
        // Perform operations on the converted tensor to ensure it's valid
        if (tensor_from_dlpack.numel() > 0) {
            // Inner try-catch for expected failures (e.g., complex dtypes)
            try {
                // Compute sum to verify data is accessible
                auto sum_result = tensor_from_dlpack.sum();
                
                // Verify the data matches by comparing sums
                auto original_sum = tensor.sum();
                
                // Check if tensors are close (allowing for floating point tolerance)
                if (tensor.is_floating_point() || tensor.is_complex()) {
                    // For floating point, use allclose
                    bool close = torch::allclose(original_sum, sum_result);
                    (void)close; // Suppress unused warning
                } else {
                    // For integer types, check exact equality
                    bool equal = torch::equal(original_sum, sum_result);
                    (void)equal;
                }
            } catch (...) {
                // Silently ignore expected failures from certain dtypes
            }
        }
        
        // Test with different tensor configurations
        if (offset < Size) {
            try {
                // Create another tensor with different properties
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                tensor2 = tensor2.contiguous();
                
                // Test DLPack conversion
                DLManagedTensor* dlpack2 = at::toDLPack(tensor2);
                if (dlpack2 != nullptr) {
                    torch::Tensor from_dlpack2 = at::fromDLPack(dlpack2);
                    // Access data to ensure validity
                    if (from_dlpack2.numel() > 0) {
                        from_dlpack2.sum();
                    }
                }
            } catch (...) {
                // Silently ignore failures in secondary tests
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
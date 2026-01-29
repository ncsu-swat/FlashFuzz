#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least some bytes for tensor creation
        if (Size < 4)
            return 0;
            
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // vdot requires 1D tensors with same number of elements
        int64_t numel1 = tensor1.numel();
        int64_t numel2 = tensor2.numel();
        
        // If either tensor has zero elements, skip
        if (numel1 == 0 || numel2 == 0) {
            return 0;
        }
        
        // Reshape tensors to 1D if they're not already
        if (tensor1.dim() != 1) {
            tensor1 = tensor1.reshape({numel1});
        }
        
        if (tensor2.dim() != 1) {
            tensor2 = tensor2.reshape({numel2});
        }
        
        // Make tensors have the same number of elements
        if (numel1 != numel2) {
            int64_t min_numel = std::min(numel1, numel2);
            tensor1 = tensor1.slice(0, 0, min_numel);
            tensor2 = tensor2.slice(0, 0, min_numel);
        }
        
        // vdot works with complex tensors (conjugates first arg) or real tensors
        // Ensure both tensors have compatible dtypes
        try {
            // Convert to same dtype if needed - use float for better compatibility
            if (tensor1.dtype() != tensor2.dtype()) {
                tensor1 = tensor1.to(torch::kFloat);
                tensor2 = tensor2.to(torch::kFloat);
            }
            
            // Apply vdot operation
            torch::Tensor result = torch::vdot(tensor1, tensor2);
            
            // Also test with contiguous tensors to exercise different code paths
            torch::Tensor result_contig = torch::vdot(tensor1.contiguous(), tensor2.contiguous());
        }
        catch (const c10::Error &e) {
            // Expected errors from incompatible types, etc - silently ignore
        }
        
        // Test with complex tensors if original tensors are floating point
        try {
            if (tensor1.is_floating_point()) {
                torch::Tensor complex1 = torch::complex(tensor1, tensor1);
                torch::Tensor complex2 = torch::complex(tensor2, tensor2);
                torch::Tensor result_complex = torch::vdot(complex1, complex2);
            }
        }
        catch (const c10::Error &e) {
            // Silently ignore expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
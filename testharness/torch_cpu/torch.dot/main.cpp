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
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Use same tensor for self dot product
            tensor2 = tensor1.clone();
        }

        // torch.dot requires 1D tensors with the same number of elements
        // Reshape tensors to 1D
        if (tensor1.dim() != 1) {
            tensor1 = tensor1.reshape(-1);
        }
        
        if (tensor2.dim() != 1) {
            tensor2 = tensor2.reshape(-1);
        }

        // Ensure same dtype (convert tensor2 to tensor1's dtype)
        if (tensor1.dtype() != tensor2.dtype()) {
            // Use inner try-catch for type conversion which may fail for some dtypes
            try {
                tensor2 = tensor2.to(tensor1.dtype());
            } catch (...) {
                // If conversion fails, try converting both to float
                try {
                    tensor1 = tensor1.to(torch::kFloat32);
                    tensor2 = tensor2.to(torch::kFloat32);
                } catch (...) {
                    return 0; // Skip if conversion not possible
                }
            }
        }

        // Inner try-catch for expected failures (size mismatch, unsupported dtype)
        try {
            // Make tensors same size by truncating to smaller size
            int64_t min_size = std::min(tensor1.size(0), tensor2.size(0));
            if (min_size > 0) {
                tensor1 = tensor1.slice(0, 0, min_size);
                tensor2 = tensor2.slice(0, 0, min_size);
                
                // Perform dot product
                torch::Tensor result = torch::dot(tensor1, tensor2);
                
                // Access result to ensure computation happens
                (void)result.item<float>();
            }
        } catch (const c10::Error &) {
            // Expected errors (unsupported dtype combinations, etc.) - silently ignore
        } catch (const std::runtime_error &) {
            // Expected runtime errors - silently ignore
        }

        // Also test with contiguous tensors
        try {
            tensor1 = tensor1.contiguous();
            tensor2 = tensor2.contiguous();
            torch::Tensor result2 = torch::dot(tensor1, tensor2);
            (void)result2.item<float>();
        } catch (const c10::Error &) {
            // Silently ignore expected errors
        } catch (const std::runtime_error &) {
            // Silently ignore expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
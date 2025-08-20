#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least two tensors for smm operation
        if (Size < 4) {
            return 0;
        }
        
        // Create the first tensor (sparse matrix)
        torch::Tensor sparse_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the second tensor (dense matrix)
        if (offset < Size) {
            torch::Tensor dense_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure sparse_tensor is 2D for smm operation
            if (sparse_tensor.dim() == 2) {
                // Try to perform the sparse matrix multiplication
                try {
                    // Convert to sparse if not already
                    torch::Tensor sparse_coo = sparse_tensor.to_sparse();
                    
                    // Perform sparse matrix multiplication
                    torch::Tensor result = torch::smm(sparse_coo, dense_tensor);
                    
                    // Verify result by comparing with dense matrix multiplication
                    if (dense_tensor.dim() == 2 && 
                        sparse_tensor.size(1) == dense_tensor.size(0)) {
                        torch::Tensor dense_result = torch::matmul(sparse_tensor, dense_tensor);
                        
                        // Check if results are close
                        bool is_close = torch::allclose(result, dense_result, 1e-4, 1e-5);
                        
                        // Use the result to prevent optimization
                        if (!is_close && result.numel() > 0) {
                            volatile float dummy = result.sum().item<float>();
                            (void)dummy;
                        }
                    }
                } catch (const c10::Error& e) {
                    // PyTorch specific errors are expected and handled
                }
            }
            
            // Try with transposed sparse matrix
            if (sparse_tensor.dim() == 2) {
                try {
                    torch::Tensor sparse_coo = sparse_tensor.to_sparse();
                    torch::Tensor transposed = sparse_coo.transpose(0, 1);
                    torch::Tensor result = torch::smm(transposed, dense_tensor);
                    
                    // Use the result to prevent optimization
                    if (result.numel() > 0) {
                        volatile float dummy = result.sum().item<float>();
                        (void)dummy;
                    }
                } catch (const c10::Error& e) {
                    // PyTorch specific errors are expected and handled
                }
            }
            
            // Try with coalesced sparse matrix
            if (sparse_tensor.dim() == 2) {
                try {
                    torch::Tensor sparse_coo = sparse_tensor.to_sparse();
                    torch::Tensor coalesced = sparse_coo.coalesce();
                    torch::Tensor result = torch::smm(coalesced, dense_tensor);
                    
                    // Use the result to prevent optimization
                    if (result.numel() > 0) {
                        volatile float dummy = result.sum().item<float>();
                        (void)dummy;
                    }
                } catch (const c10::Error& e) {
                    // PyTorch specific errors are expected and handled
                }
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
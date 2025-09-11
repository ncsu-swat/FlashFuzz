#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 2 tensors for matmul
        if (Size < 4) {  // Minimum bytes needed for basic tensor creation
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if there's data left
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to perform matmul operation
            try {
                torch::Tensor result = torch::matmul(tensor1, tensor2);
            } 
            catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
                // This is normal when tensors have incompatible shapes
            }
            
            // Try batched matmul with different batch dimensions if possible
            if (offset < Size && Size - offset >= 1) {
                uint8_t batch_mode = Data[offset++] % 3;
                
                try {
                    if (batch_mode == 0 && tensor1.dim() >= 3 && tensor2.dim() >= 3) {
                        // Try batched matmul with explicit batch dimensions
                        torch::Tensor result = torch::matmul(tensor1, tensor2);
                    }
                    else if (batch_mode == 1) {
                        // Try broadcasting batch dimensions
                        if (tensor1.dim() >= 1 && tensor2.dim() >= 1) {
                            // Add a batch dimension to one tensor
                            torch::Tensor batched1 = tensor1.unsqueeze(0);
                            torch::Tensor result = torch::matmul(batched1, tensor2);
                        }
                    }
                    else if (batch_mode == 2) {
                        // Try with different batch dimensions
                        if (tensor1.dim() >= 2 && tensor2.dim() >= 2) {
                            // Add different batch dimensions
                            torch::Tensor batched1 = tensor1.unsqueeze(0);
                            torch::Tensor batched2 = tensor2.unsqueeze(0).unsqueeze(0);
                            torch::Tensor result = torch::matmul(batched1, batched2);
                        }
                    }
                }
                catch (const c10::Error& e) {
                    // Expected exceptions from PyTorch operations are fine
                }
            }
            
            // Try edge cases: vector-vector, matrix-vector, vector-matrix
            try {
                if (tensor1.dim() == 1 && tensor2.dim() == 1) {
                    // Vector-vector dot product
                    torch::Tensor result = torch::matmul(tensor1, tensor2);
                }
                else if (tensor1.dim() == 2 && tensor2.dim() == 1) {
                    // Matrix-vector product
                    torch::Tensor result = torch::matmul(tensor1, tensor2);
                }
                else if (tensor1.dim() == 1 && tensor2.dim() == 2) {
                    // Vector-matrix product
                    torch::Tensor result = torch::matmul(tensor1, tensor2);
                }
            }
            catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with empty tensors if possible
        if (offset < Size) {
            try {
                // Create empty tensors with compatible shapes for matmul
                torch::Tensor empty1 = torch::empty({0, 2});
                torch::Tensor empty2 = torch::empty({2, 3});
                torch::Tensor result = torch::matmul(empty1, empty2);
            }
            catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with scalar tensors if possible
        if (offset < Size) {
            try {
                torch::Tensor scalar1 = torch::tensor(3.14);
                torch::Tensor scalar2 = torch::tensor(2.71);
                torch::Tensor result = torch::matmul(scalar1, scalar2);
            }
            catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
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

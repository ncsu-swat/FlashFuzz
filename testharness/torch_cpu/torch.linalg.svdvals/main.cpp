#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // svdvals requires at least a 2D tensor with float or complex dtype
        if (A.dim() < 2) {
            return 0;
        }
        
        // Ensure the tensor has a floating point dtype
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat);
        }
        
        // Try to compute the singular values using torch::linalg_svdvals
        torch::Tensor S = torch::linalg_svdvals(A);
        
        // Try with different driver options
        if (offset + 1 < Size) {
            bool use_gesvd = Data[offset++] % 2 == 0;
            try {
                torch::Tensor S_with_driver = torch::linalg_svdvals(A, use_gesvd ? "gesvd" : "gesdd");
            } catch (...) {
                // Driver option might not be available for all inputs
            }
        }
        
        // Try with complex tensors if we have enough data
        if (offset + 4 < Size) {
            try {
                torch::Tensor real_part = A;
                torch::Tensor imag_part = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure imag_part has compatible shape and dtype
                if (imag_part.dim() >= 2) {
                    if (!imag_part.is_floating_point()) {
                        imag_part = imag_part.to(torch::kFloat);
                    }
                    // Match shapes for complex tensor creation
                    if (real_part.sizes() == imag_part.sizes() && 
                        real_part.dtype() == imag_part.dtype()) {
                        torch::Tensor complex_tensor = torch::complex(real_part, imag_part);
                        torch::Tensor complex_svd = torch::linalg_svdvals(complex_tensor);
                    }
                }
            } catch (...) {
                // Complex tensor creation might fail, that's okay
            }
        }
        
        // Try with non-square matrices if we have enough data
        if (offset + 4 < Size) {
            try {
                torch::Tensor non_square = fuzzer_utils::createTensor(Data, Size, offset);
                if (non_square.dim() >= 2) {
                    if (!non_square.is_floating_point()) {
                        non_square = non_square.to(torch::kFloat);
                    }
                    torch::Tensor non_square_svd = torch::linalg_svdvals(non_square);
                }
            } catch (...) {
                // This might fail, that's okay
            }
        }
        
        // Try with batched tensors if we have enough data
        if (A.dim() >= 2) {
            try {
                // Create a batched version by adding a batch dimension using unsqueeze and repeat
                torch::Tensor batched = A.unsqueeze(0).repeat({2, 1, 1});
                torch::Tensor batched_svd = torch::linalg_svdvals(batched);
            } catch (...) {
                // This might fail, that's okay
            }
        }
        
        // Try with double precision
        try {
            torch::Tensor A_double = A.to(torch::kDouble);
            torch::Tensor S_double = torch::linalg_svdvals(A_double);
        } catch (...) {
            // Might fail for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
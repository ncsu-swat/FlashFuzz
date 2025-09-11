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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to compute the singular values
        torch::Tensor S = torch::linalg_svdvals(A);
        
        // Try some variants with options
        if (offset + 1 < Size) {
            bool driver_option = Data[offset++] % 2 == 0;
            torch::Tensor S_with_driver = torch::linalg_svdvals(A, driver_option ? "gesvd" : "gesdd");
        }
        
        // Try with complex tensors if we have enough data
        if (offset + 4 < Size) {
            // Create a complex tensor by combining real and imaginary parts
            torch::Tensor real_part = A;
            torch::Tensor imag_part = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to make a complex tensor if possible
            try {
                torch::Tensor complex_tensor;
                if (real_part.dtype() == torch::kFloat) {
                    complex_tensor = torch::complex(real_part, imag_part);
                    torch::Tensor complex_svd = torch::linalg_svdvals(complex_tensor);
                } else if (real_part.dtype() == torch::kDouble) {
                    complex_tensor = torch::complex(real_part, imag_part);
                    torch::Tensor complex_svd = torch::linalg_svdvals(complex_tensor);
                }
            } catch (...) {
                // Complex tensor creation might fail, that's okay
            }
        }
        
        // Try with non-square matrices if we have enough data
        if (offset + 4 < Size) {
            try {
                torch::Tensor non_square = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor non_square_svd = torch::linalg_svdvals(non_square);
            } catch (...) {
                // This might fail, that's okay
            }
        }
        
        // Try with batched tensors if we have enough data
        if (offset + 4 < Size && A.dim() >= 2) {
            try {
                // Create a batched version by adding a batch dimension
                std::vector<int64_t> batch_shape = A.sizes().vec();
                batch_shape.insert(batch_shape.begin(), 2); // Add batch dimension of size 2
                
                torch::Tensor batched = A.expand(batch_shape);
                torch::Tensor batched_svd = torch::linalg_svdvals(batched);
            } catch (...) {
                // This might fail, that's okay
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

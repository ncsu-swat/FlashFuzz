#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a square matrix for eigvalsh
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // eigvalsh requires a square matrix, so we need to make sure our tensor is square
        // We'll reshape it if needed
        if (A.dim() < 2) {
            // If tensor is 0D or 1D, reshape to a small square matrix
            int64_t size = 2;
            if (A.numel() > 4) {
                size = 3;
            }
            
            // Ensure we have enough elements
            if (A.numel() > 0) {
                // Repeat the tensor to get enough elements if needed
                while (A.numel() < size * size) {
                    A = A.repeat(2);
                }
                A = A.reshape({size, size});
            } else {
                // Create a small random tensor if we have an empty tensor
                A = torch::randn({size, size});
            }
        } else if (A.dim() > 2) {
            // If tensor has more than 2 dimensions, take the first 2 dimensions
            std::vector<int64_t> new_shape = {A.size(0), A.size(1)};
            int64_t total_elements = A.size(0) * A.size(1);
            
            // Ensure we have enough elements
            if (A.numel() >= total_elements) {
                A = A.flatten().slice(0, 0, total_elements).reshape(new_shape);
            } else {
                // Not enough elements, create a new tensor
                A = torch::randn(new_shape);
            }
        }
        
        // Make sure the matrix is square
        int64_t min_dim = std::min(A.size(0), A.size(1));
        A = A.slice(0, 0, min_dim).slice(1, 0, min_dim);
        
        // Make the matrix Hermitian/symmetric for eigvalsh
        // For real matrices: A = 0.5 * (A + A.t())
        // For complex matrices: A = 0.5 * (A + A.conj().transpose(-2, -1))
        if (A.is_complex()) {
            A = 0.5 * (A + A.conj().transpose(-2, -1));
        } else {
            A = 0.5 * (A + A.t());
        }
        
        // Get a byte for UPLO parameter
        char uplo = 'L'; // Default to lower
        if (offset < Size) {
            uplo = (Data[offset++] % 2 == 0) ? 'L' : 'U';
        }
        
        // Try different eigvalsh variants
        try {
            // Basic eigvalsh call
            torch::Tensor eigenvalues = torch::symeig(A, false).values;
        } catch (const c10::Error& e) {
            // PyTorch specific error, continue with other variants
        }
        
        try {
            // eigvalsh with UPLO parameter - using symeig as alternative
            torch::Tensor eigenvalues = torch::symeig(A, false, uplo == 'U').values;
        } catch (const c10::Error& e) {
            // PyTorch specific error, continue with other variants
        }
        
        // Try with different dtypes if possible
        if (offset < Size) {
            try {
                auto dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Convert A to the new dtype if possible
                torch::Tensor A_converted;
                try {
                    A_converted = A.to(dtype);
                    torch::Tensor eigenvalues = torch::symeig(A_converted, false).values;
                } catch (const c10::Error& e) {
                    // Type conversion or eigvalsh failed, which is expected for some dtypes
                }
            } catch (const std::exception& e) {
                // Ignore dtype conversion errors
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
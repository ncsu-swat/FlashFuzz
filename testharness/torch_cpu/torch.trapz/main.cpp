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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor y
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have enough data, create a second tensor x for the coordinates
        torch::Tensor x;
        if (offset + 4 < Size) {
            x = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get a dimension to integrate over
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different variants of trapz
        try {
            // Basic trapz with default parameters
            torch::Tensor result1 = torch::trapz(y);
            
            // Trapz with dimension specified
            if (y.dim() > 0) {
                dim = dim % y.dim();
                torch::Tensor result2 = torch::trapz(y, dim);
            }
            
            // Trapz with x coordinates and dimension
            if (x.defined() && y.dim() > 0) {
                dim = dim % y.dim();
                
                // Try to make x compatible with y for the given dimension
                if (x.dim() == 1) {
                    // If x is 1D, it should match the size of y along dim
                    if (x.size(0) == y.size(dim)) {
                        torch::Tensor result3 = torch::trapz(y, x, dim);
                    }
                }
                
                // Try with a scalar dx value
                double dx = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&dx, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                
                torch::Tensor result4 = torch::trapz(y, dx, dim);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but continue fuzzing
        }
        
        // Try edge cases
        try {
            // Empty tensor
            if (y.numel() == 0) {
                torch::Tensor result_empty = torch::trapz(y);
            }
            
            // 0-dimensional tensor (scalar)
            if (y.dim() == 0) {
                torch::Tensor result_scalar = torch::trapz(y);
            }
            
            // 1-dimensional tensor
            if (y.dim() == 1) {
                torch::Tensor result_1d = torch::trapz(y);
            }
            
            // Negative dimension
            if (y.dim() > 0) {
                int64_t neg_dim = -1;
                torch::Tensor result_neg_dim = torch::trapz(y, neg_dim);
            }
            
            // Very small dx
            if (y.dim() > 0) {
                double tiny_dx = 1e-10;
                torch::Tensor result_tiny_dx = torch::trapz(y, tiny_dx, 0);
            }
            
            // Very large dx
            if (y.dim() > 0) {
                double huge_dx = 1e10;
                torch::Tensor result_huge_dx = torch::trapz(y, huge_dx, 0);
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but continue fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

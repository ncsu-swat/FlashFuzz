#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Parse matrix dimensions for a 2D matrix
        uint8_t dim0_raw = Data[offset++];
        uint8_t dim1_raw = Data[offset++];
        
        // Ensure reasonable matrix dimensions (at least 1x1, at most 64x64)
        int64_t dim0 = (dim0_raw % 64) + 1;
        int64_t dim1 = (dim1_raw % 64) + 1;
        
        // Parse p-norm parameter
        uint8_t p_selector = Data[offset++];
        
        // Parse dtype
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype;
        switch (dtype_selector % 3) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create a 2D matrix tensor
        torch::Tensor A;
        
        // Use remaining data to seed the tensor or create random
        if (offset + sizeof(float) * 4 <= Size) {
            A = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape to 2D if needed
            if (A.dim() != 2) {
                int64_t total = A.numel();
                if (total < dim0 * dim1) {
                    // Pad with random values
                    A = torch::randn({dim0, dim1}, torch::dtype(dtype));
                } else {
                    A = A.flatten().slice(0, 0, dim0 * dim1).reshape({dim0, dim1});
                }
            }
        } else {
            A = torch::randn({dim0, dim1}, torch::dtype(dtype));
        }
        
        // Ensure correct dtype
        A = A.to(dtype);
        
        // Select p-norm based on input data
        // linalg_cond supports: None/'fro', 'nuc', 1, -1, 2, -2, inf, -inf
        c10::optional<c10::Scalar> p_norm = c10::nullopt;
        
        switch (p_selector % 8) {
            case 0:
                p_norm = c10::nullopt; // Default (2-norm for matrices)
                break;
            case 1:
                p_norm = 1;
                break;
            case 2:
                p_norm = -1;
                break;
            case 3:
                p_norm = 2;
                break;
            case 4:
                p_norm = -2;
                break;
            case 5:
                p_norm = std::numeric_limits<double>::infinity();
                break;
            case 6:
                p_norm = -std::numeric_limits<double>::infinity();
                break;
            case 7:
                // Frobenius norm (use string "fro" equivalent - in C++ API, use nullopt or specific call)
                p_norm = c10::nullopt;
                break;
        }
        
        // Inner try-catch for expected failures (shape mismatches, singular matrices, etc.)
        try {
            torch::Tensor result = torch::linalg_cond(A, p_norm);
            
            // Access the result to ensure computation is performed
            if (result.defined() && result.numel() > 0) {
                // Sum to handle both scalar and batched results
                volatile float value = result.sum().item<float>();
                (void)value;
            }
        } catch (const c10::Error&) {
            // Expected failures (e.g., singular matrix, incompatible shapes)
            // Silently ignore
        } catch (const std::runtime_error&) {
            // Expected runtime errors
            // Silently ignore
        }
        
        // Also test with square matrices (required for some norms)
        if (dim0 != dim1) {
            int64_t square_dim = std::min(dim0, dim1);
            torch::Tensor A_square = torch::randn({square_dim, square_dim}, torch::dtype(dtype));
            
            try {
                torch::Tensor result_square = torch::linalg_cond(A_square, p_norm);
                if (result_square.defined() && result_square.numel() > 0) {
                    volatile float value = result_square.sum().item<float>();
                    (void)value;
                }
            } catch (const c10::Error&) {
                // Expected failures
            } catch (const std::runtime_error&) {
                // Expected failures
            }
        }
        
        // Test batched input
        try {
            int64_t batch_size = (p_selector % 3) + 1;
            int64_t sq_dim = std::min(dim0, dim1);
            torch::Tensor A_batched = torch::randn({batch_size, sq_dim, sq_dim}, torch::dtype(dtype));
            
            torch::Tensor result_batched = torch::linalg_cond(A_batched, p_norm);
            if (result_batched.defined() && result_batched.numel() > 0) {
                volatile float value = result_batched.sum().item<float>();
                (void)value;
            }
        } catch (const c10::Error&) {
            // Expected failures
        } catch (const std::runtime_error&) {
            // Expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
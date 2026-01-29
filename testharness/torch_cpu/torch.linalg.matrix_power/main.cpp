#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

// Helper function to compute matrix power
torch::Tensor compute_matrix_power(const torch::Tensor& A, int64_t n) {
    int64_t dim = A.size(-1);
    
    if (n == 0) {
        // Return identity matrix (or batch of identity matrices)
        auto sizes = A.sizes().vec();
        sizes[sizes.size() - 2] = dim;
        sizes[sizes.size() - 1] = dim;
        return torch::eye(dim, A.options()).expand(sizes).contiguous();
    }
    
    bool negative = n < 0;
    if (negative) {
        n = -n;
    }
    
    torch::Tensor base = A;
    if (negative) {
        // Compute inverse for negative powers
        base = torch::inverse(A);
    }
    
    // Binary exponentiation
    torch::Tensor result = torch::eye(dim, A.options());
    if (A.dim() > 2) {
        // Handle batched case
        auto sizes = A.sizes().vec();
        result = result.expand(sizes).contiguous();
    }
    
    while (n > 0) {
        if (n % 2 == 1) {
            result = torch::matmul(result, base);
        }
        base = torch::matmul(base, base);
        n /= 2;
    }
    
    return result;
}

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
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from fuzzer data
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);

        // Ensure we have a valid square matrix
        int64_t elements = A.numel();
        if (elements == 0) {
            return 0;
        }

        // Calculate a reasonable square dimension
        int64_t dim_size = static_cast<int64_t>(std::sqrt(static_cast<double>(elements)));
        if (dim_size < 1) {
            dim_size = 1;
        }
        // Limit matrix size to avoid excessive computation
        if (dim_size > 32) {
            dim_size = 32;
        }

        // Create a square matrix by taking first dim_size*dim_size elements
        int64_t needed = dim_size * dim_size;
        torch::Tensor flat = A.flatten();
        if (flat.numel() < needed) {
            // Pad with zeros if needed
            flat = torch::cat({flat, torch::zeros({needed - flat.numel()}, flat.options())});
        }
        A = flat.slice(0, 0, needed).reshape({dim_size, dim_size});

        // Ensure floating point type for matrix operations
        if (!A.is_floating_point()) {
            A = A.to(torch::kFloat);
        }

        // Extract n (power) from the input data, limit range to avoid huge computations
        int32_t n = 0;
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Limit power to reasonable range
            n = n % 21 - 10;  // Range: -10 to 10
        }

        // Apply matrix_power operation
        torch::Tensor result;
        try {
            result = compute_matrix_power(A, static_cast<int64_t>(n));
        } catch (const std::exception &) {
            // Negative powers on singular matrices will fail - expected
        }

        // Test with different data types
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            try {
                if (variant == 0) {
                    // Test with double precision
                    torch::Tensor A_double = A.to(torch::kDouble);
                    result = compute_matrix_power(A_double, static_cast<int64_t>(n));
                } else if (variant == 1) {
                    // Test with complex float
                    torch::Tensor A_complex = A.to(torch::kComplexFloat);
                    result = compute_matrix_power(A_complex, static_cast<int64_t>(n));
                } else {
                    // Test with batch dimension
                    torch::Tensor A_batch = A.unsqueeze(0).expand({2, dim_size, dim_size}).contiguous();
                    result = compute_matrix_power(A_batch, static_cast<int64_t>(n));
                }
            } catch (const std::exception &) {
                // Type conversions or operations may fail - expected
            }
        }

        // Test edge cases with specific powers
        if (offset < Size) {
            uint8_t power_case = Data[offset++] % 5;
            try {
                switch (power_case) {
                    case 0:
                        // Power 0 (should return identity matrix)
                        result = compute_matrix_power(A, 0);
                        break;
                    case 1:
                        // Power 1 (should return A)
                        result = compute_matrix_power(A, 1);
                        break;
                    case 2:
                        // Power 2
                        result = compute_matrix_power(A, 2);
                        break;
                    case 3:
                        // Negative power (requires invertible matrix)
                        result = compute_matrix_power(A, -1);
                        break;
                    case 4:
                        // Negative power -2
                        result = compute_matrix_power(A, -2);
                        break;
                }
            } catch (const std::exception &) {
                // Singular matrices with negative power will fail - expected
            }
        }

        // Test with identity-like matrix to ensure negative powers work sometimes
        if (offset < Size && (Data[offset] % 4 == 0)) {
            try {
                torch::Tensor eye = torch::eye(dim_size, A.options());
                // Add small perturbation to make it more interesting
                float max_val = A.abs().max().item<float>() + 1e-6f;
                eye = eye + 0.1f * A / max_val;
                result = compute_matrix_power(eye, -3);
            } catch (const std::exception &) {
                // May still fail if perturbation makes it singular
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
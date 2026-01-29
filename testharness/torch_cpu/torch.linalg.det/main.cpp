#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::sqrt

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor for determinant calculation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determinant requires a square matrix (..., n, n)
        // Handle different input dimensions
        int64_t total_elements = input.numel();
        
        if (total_elements < 1) {
            return 0;
        }
        
        // Find a suitable square size
        int64_t square_size = static_cast<int64_t>(std::sqrt(static_cast<double>(total_elements)));
        if (square_size < 1) {
            square_size = 1;
        }
        
        // Flatten and take required elements for a square matrix
        input = input.flatten().slice(0, 0, square_size * square_size);
        input = input.reshape({square_size, square_size});
        
        // Convert to a supported dtype for determinant (needs floating point or complex)
        if (input.scalar_type() == torch::kBool || 
            input.scalar_type() == torch::kUInt8 || 
            input.scalar_type() == torch::kInt8 || 
            input.scalar_type() == torch::kInt16 || 
            input.scalar_type() == torch::kInt32 || 
            input.scalar_type() == torch::kInt64) {
            input = input.to(torch::kFloat);
        }
        
        // Calculate determinant using torch::linalg_det (the API under test)
        torch::Tensor det = torch::linalg_det(input);
        
        // Also test batched determinant if we have enough data
        if (offset + 2 < Size) {
            uint8_t batch_size = (Data[offset++] % 4) + 1; // 1-4 batches
            try {
                torch::Tensor batched_input = input.unsqueeze(0).expand({batch_size, square_size, square_size}).clone();
                torch::Tensor batched_det = torch::linalg_det(batched_input);
            } catch (...) {
                // Silently ignore batched edge cases
            }
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                // Edge case: matrix with zeros (singular)
                if (edge_case % 6 == 0) {
                    torch::Tensor zero_matrix = torch::zeros_like(input);
                    torch::Tensor zero_det = torch::linalg_det(zero_matrix);
                }
                
                // Edge case: identity matrix (det = 1)
                if (edge_case % 6 == 1) {
                    torch::Tensor identity = torch::eye(input.size(0), input.options());
                    torch::Tensor identity_det = torch::linalg_det(identity);
                }
                
                // Edge case: matrix with very large values
                if (edge_case % 6 == 2) {
                    torch::Tensor large_matrix = input * 1e10;
                    torch::Tensor large_det = torch::linalg_det(large_matrix);
                }
                
                // Edge case: matrix with very small values
                if (edge_case % 6 == 3) {
                    torch::Tensor small_matrix = input * 1e-10;
                    torch::Tensor small_det = torch::linalg_det(small_matrix);
                }
                
                // Edge case: singular matrix (make rows linearly dependent)
                if (edge_case % 6 == 4 && input.size(0) > 1) {
                    torch::Tensor singular = input.clone();
                    singular.index_put_({1}, singular.index({0}));
                    torch::Tensor singular_det = torch::linalg_det(singular);
                }
                
                // Edge case: diagonal matrix
                if (edge_case % 6 == 5) {
                    torch::Tensor diag_values = input.diagonal();
                    torch::Tensor diag_matrix = torch::diag(diag_values);
                    torch::Tensor diag_det = torch::linalg_det(diag_matrix);
                }
            } catch (...) {
                // Silently ignore edge case exceptions (expected for some inputs)
            }
        }
        
        // Test with different dtypes if we have data
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            try {
                if (dtype_choice == 0) {
                    // Test with double precision
                    torch::Tensor double_input = input.to(torch::kDouble);
                    torch::Tensor double_det = torch::linalg_det(double_input);
                } else if (dtype_choice == 1) {
                    // Test with complex float
                    torch::Tensor complex_input = torch::complex(input, torch::zeros_like(input));
                    torch::Tensor complex_det = torch::linalg_det(complex_input);
                } else {
                    // Test with complex double
                    torch::Tensor input_double = input.to(torch::kDouble);
                    torch::Tensor complex_input = torch::complex(input_double, torch::zeros_like(input_double));
                    torch::Tensor complex_det = torch::linalg_det(complex_input);
                }
            } catch (...) {
                // Silently ignore dtype conversion edge cases
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}
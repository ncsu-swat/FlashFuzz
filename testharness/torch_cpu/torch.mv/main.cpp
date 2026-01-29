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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input matrix (2D tensor)
        torch::Tensor mat = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we don't have enough data for the second tensor, return
        if (offset >= Size) {
            return 0;
        }
        
        // Create input vector (1D tensor)
        torch::Tensor vec = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to reshape tensors if needed to make them compatible with mv
        // Matrix should be 2D
        if (mat.dim() != 2) {
            if (mat.dim() == 0) {
                mat = mat.reshape({1, 1});
            } else if (mat.dim() == 1) {
                mat = mat.reshape({1, mat.size(0)});
            } else {
                // For higher dimensions, flatten all but the last dimension
                std::vector<int64_t> new_shape = {1, -1};
                mat = mat.reshape(new_shape);
            }
        }
        
        // Vector should be 1D
        if (vec.dim() != 1) {
            if (vec.dim() == 0) {
                vec = vec.reshape({1});
            } else {
                // For higher dimensions, flatten to 1D
                vec = vec.reshape({-1});
            }
        }
        
        // Try to make dimensions compatible if possible
        // For mv: mat.size(1) should match vec.size(0)
        if (mat.size(1) != vec.size(0)) {
            // If vector is too short, pad it
            if (vec.size(0) < mat.size(1)) {
                torch::Tensor padded_vec = torch::zeros({mat.size(1)}, vec.options());
                padded_vec.index_put_({torch::indexing::Slice(0, vec.size(0))}, vec);
                vec = padded_vec;
            }
            // If vector is too long, truncate it
            else if (vec.size(0) > mat.size(1)) {
                vec = vec.index({torch::indexing::Slice(0, mat.size(1))});
            }
        }
        
        // Ensure compatible dtypes
        if (mat.scalar_type() != vec.scalar_type()) {
            // Convert to float for simplicity
            mat = mat.to(torch::kFloat);
            vec = vec.to(torch::kFloat);
        }
        
        // Apply the mv operation
        torch::Tensor result = torch::mv(mat, vec);
        
        // Optional: Test edge cases with different dimensions
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Test with empty tensors
            if (edge_case % 4 == 0 && mat.size(0) > 0 && mat.size(1) > 0) {
                torch::Tensor empty_mat = torch::empty({mat.size(0), 0}, mat.options());
                torch::Tensor empty_vec = torch::empty({0}, vec.options());
                try {
                    torch::Tensor empty_result = torch::mv(empty_mat, empty_vec);
                } catch (...) {
                    // Expected exception for incompatible dimensions
                }
            }
            
            // Test with 1x1 matrix and 1-element vector
            if (edge_case % 4 == 1) {
                torch::Tensor small_mat = torch::ones({1, 1}, mat.options());
                torch::Tensor small_vec = torch::ones({1}, vec.options());
                torch::Tensor small_result = torch::mv(small_mat, small_vec);
            }
            
            // Test with very large values
            if (edge_case % 4 == 2) {
                torch::Tensor large_mat = torch::full({2, 2}, 1e10, mat.options());
                torch::Tensor large_vec = torch::full({2}, 1e10, vec.options());
                try {
                    torch::Tensor large_result = torch::mv(large_mat, large_vec);
                } catch (...) {
                    // May throw for overflow
                }
            }
            
            // Test with NaN/Inf values for floating point types
            if (edge_case % 4 == 3 && 
                (mat.scalar_type() == torch::kFloat || 
                 mat.scalar_type() == torch::kDouble || 
                 mat.scalar_type() == torch::kHalf)) {
                torch::Tensor special_mat = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), mat.options());
                torch::Tensor special_vec = torch::full({2}, std::numeric_limits<float>::infinity(), vec.options());
                try {
                    torch::Tensor special_result = torch::mv(special_mat, special_vec);
                } catch (...) {
                    // May throw for NaN/Inf
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
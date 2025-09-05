#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper function to make a matrix positive-definite
torch::Tensor makePositiveDefinite(torch::Tensor A) {
    // For a matrix to be positive-definite, we can use A @ A.T + eps * I
    // This guarantees positive eigenvalues
    torch::Tensor result;
    
    if (A.is_complex()) {
        // For complex matrices, use conjugate transpose
        result = torch::matmul(A, A.conj().transpose(-2, -1));
    } else {
        // For real matrices, use regular transpose
        result = torch::matmul(A, A.transpose(-2, -1));
    }
    
    // Add a small diagonal to ensure positive definiteness
    auto eye = torch::eye(A.size(-1), A.options());
    
    // Expand eye for batched operations if needed
    if (A.dim() > 2) {
        auto batch_shape = A.sizes().vec();
        batch_shape.pop_back();
        batch_shape.pop_back();
        batch_shape.push_back(A.size(-2));
        batch_shape.push_back(A.size(-1));
        eye = eye.expand(batch_shape);
    }
    
    // Add scaled identity to ensure positive definiteness
    result = result + eye * 0.1;
    
    return result;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4) {
        return 0; // Need minimum bytes for configuration
    }
    
    try {
        size_t offset = 0;
        
        // Parse configuration bytes
        uint8_t config_byte = data[offset++];
        bool use_upper = (config_byte & 0x01) != 0;
        bool use_batched = (config_byte & 0x02) != 0;
        bool make_pd = (config_byte & 0x04) != 0; // Whether to make positive-definite
        bool use_complex = (config_byte & 0x08) != 0;
        
        // Determine dtype based on complex flag and another byte
        uint8_t dtype_selector = data[offset++];
        torch::ScalarType dtype;
        
        if (use_complex) {
            dtype = (dtype_selector % 2 == 0) ? torch::kComplexFloat : torch::kComplexDouble;
        } else {
            dtype = (dtype_selector % 2 == 0) ? torch::kFloat : torch::kDouble;
        }
        
        // Parse dimensions
        uint8_t n_byte = data[offset++];
        int64_t n = (n_byte % 8) + 2; // Matrix size between 2 and 9
        
        std::vector<int64_t> shape;
        
        if (use_batched && offset < size) {
            uint8_t batch_byte = data[offset++];
            int64_t batch_size = (batch_byte % 4) + 1; // Batch size 1-4
            
            // Support multi-dimensional batches
            if (batch_byte > 127 && offset < size) {
                uint8_t batch2_byte = data[offset++];
                int64_t batch_size2 = (batch2_byte % 3) + 1; // Additional batch dimension 1-3
                shape.push_back(batch_size2);
            }
            
            shape.push_back(batch_size);
        }
        
        shape.push_back(n);
        shape.push_back(n);
        
        // Create input tensor
        torch::Tensor A;
        
        if (offset < size && (size - offset) >= 2) {
            // Try to create tensor from remaining data
            try {
                A = fuzzer_utils::createTensor(data, size, offset);
                
                // Reshape to our desired shape if possible
                int64_t total_elements = 1;
                for (auto dim : shape) {
                    total_elements *= dim;
                }
                
                if (A.numel() >= total_elements) {
                    A = A.narrow(0, 0, total_elements).reshape(shape);
                } else {
                    // Not enough elements, create random tensor
                    A = torch::randn(shape, torch::TensorOptions().dtype(dtype));
                }
                
                // Ensure correct dtype
                A = A.to(dtype);
            } catch (...) {
                // If tensor creation fails, use random
                A = torch::randn(shape, torch::TensorOptions().dtype(dtype));
            }
        } else {
            // Create random tensor
            A = torch::randn(shape, torch::TensorOptions().dtype(dtype));
        }
        
        // Make the matrix positive-definite if requested or with some probability
        if (make_pd || (data[0] % 3 != 0)) { // 2/3 chance of making PD
            A = makePositiveDefinite(A);
        }
        
        // Test the Cholesky decomposition
        try {
            torch::Tensor L = torch::linalg::cholesky(A, use_upper);
            
            // Verify the decomposition if we have a valid result
            if (L.defined() && L.numel() > 0) {
                // Reconstruct the original matrix
                torch::Tensor reconstructed;
                if (use_upper) {
                    if (L.is_complex()) {
                        reconstructed = torch::matmul(L.conj().transpose(-2, -1), L);
                    } else {
                        reconstructed = torch::matmul(L.transpose(-2, -1), L);
                    }
                } else {
                    if (L.is_complex()) {
                        reconstructed = torch::matmul(L, L.conj().transpose(-2, -1));
                    } else {
                        reconstructed = torch::matmul(L, L.transpose(-2, -1));
                    }
                }
                
                // Check if reconstruction is close to original (for PD matrices)
                if (make_pd) {
                    bool close = torch::allclose(reconstructed, A, 1e-3, 1e-5);
                    if (!close && A.numel() < 100) {
                        // Log large discrepancies for small matrices
                        auto max_diff = torch::max(torch::abs(reconstructed - A)).item<double>();
                        if (max_diff > 0.1) {
                            std::cerr << "Large reconstruction error: " << max_diff << std::endl;
                        }
                    }
                }
                
                // Check triangular property
                if (use_upper) {
                    // Check that L is upper triangular
                    auto lower_part = torch::tril(L, -1);
                    auto max_lower = torch::max(torch::abs(lower_part)).item<double>();
                    if (max_lower > 1e-6) {
                        std::cerr << "Upper triangular check failed: " << max_lower << std::endl;
                    }
                } else {
                    // Check that L is lower triangular
                    auto upper_part = torch::triu(L, 1);
                    auto max_upper = torch::max(torch::abs(upper_part)).item<double>();
                    if (max_upper > 1e-6) {
                        std::cerr << "Lower triangular check failed: " << max_upper << std::endl;
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // Expected errors for non-positive-definite matrices
            // This is normal behavior - the function should throw for invalid inputs
            std::string error_msg = e.what();
            if (error_msg.find("positive-definite") != std::string::npos ||
                error_msg.find("Cholesky") != std::string::npos) {
                // Expected error for non-PD matrix
                return 0;
            }
            // Unexpected error
            std::cerr << "Unexpected c10::Error: " << e.what() << std::endl;
            return 0;
        }
        
        // Test with output tensor
        if (offset < size && data[offset] % 4 == 0) {
            torch::Tensor out = torch::empty_like(A);
            try {
                torch::linalg::cholesky_out(out, A, use_upper);
                
                // Verify out was properly filled
                if (!out.isnan().any().item<bool>()) {
                    // Success
                }
            } catch (const c10::Error& e) {
                // Expected for non-PD matrices
                return 0;
            }
        }
        
        // Test edge cases
        if (shape.back() == 2 && offset < size && data[offset] % 10 == 0) {
            // Test with singular matrix (not positive-definite)
            torch::Tensor singular = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
            try {
                torch::linalg::cholesky(singular, use_upper);
                // Should have thrown
                std::cerr << "Failed to throw for singular matrix" << std::endl;
            } catch (const c10::Error& e) {
                // Expected
            }
        }
        
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}
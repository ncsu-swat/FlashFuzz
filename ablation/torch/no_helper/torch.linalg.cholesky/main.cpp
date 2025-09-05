#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for configuration
    }

    try {
        // Parse configuration from fuzzer input
        size_t offset = 0;
        
        // Determine dtype (4 options: float, double, cfloat, cdouble)
        uint8_t dtype_choice = data[offset++] % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat;
        }
        
        // Determine upper flag
        bool upper = data[offset++] % 2;
        
        // Determine number of batch dimensions (0-3)
        uint8_t num_batch_dims = data[offset++] % 4;
        
        // Determine matrix size n (1-32 to keep memory reasonable)
        uint8_t n = (data[offset++] % 32) + 1;
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < num_batch_dims; i++) {
            if (offset >= size) break;
            uint8_t batch_size = (data[offset++] % 4) + 1;  // batch sizes 1-4
            shape.push_back(batch_size);
        }
        shape.push_back(n);
        shape.push_back(n);
        
        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Create options for tensor
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Strategy selection for tensor creation
        uint8_t strategy = (offset < size) ? data[offset++] % 5 : 0;
        
        torch::Tensor A;
        
        switch (strategy) {
            case 0: {
                // Random positive-definite matrix
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    // Complex case: A = B @ B.H + I
                    torch::Tensor B = torch::randn(shape, options);
                    torch::Tensor I = torch::eye(n, options).expand(shape);
                    A = torch::matmul(B, B.conj().transpose(-2, -1)) + I;
                } else {
                    // Real case: A = B @ B.T + I
                    torch::Tensor B = torch::randn(shape, options);
                    torch::Tensor I = torch::eye(n, options).expand(shape);
                    A = torch::matmul(B, B.transpose(-2, -1)) + I;
                }
                break;
            }
            case 1: {
                // Identity matrix (always positive-definite)
                A = torch::eye(n, options).expand(shape);
                break;
            }
            case 2: {
                // Diagonal matrix with positive values
                torch::Tensor diag_values = torch::rand({n}, options.dtype(torch::kFloat)) + 0.1;
                A = torch::diag(diag_values).to(dtype).expand(shape);
                break;
            }
            case 3: {
                // Parse raw bytes as tensor values (may not be positive-definite)
                size_t bytes_per_element = (dtype == torch::kFloat || dtype == torch::kComplexFloat) ? 4 : 8;
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    bytes_per_element *= 2;  // Complex numbers need twice the space
                }
                
                size_t bytes_needed = total_elements * bytes_per_element;
                if (offset + bytes_needed <= size) {
                    A = torch::from_blob((void*)(data + offset), shape, options).clone();
                    offset += bytes_needed;
                } else {
                    // Not enough data, fall back to random
                    A = torch::randn(shape, options);
                }
                
                // Try to make it positive-definite
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    A = torch::matmul(A, A.conj().transpose(-2, -1)) + torch::eye(n, options).expand(shape) * 0.1;
                } else {
                    A = torch::matmul(A, A.transpose(-2, -1)) + torch::eye(n, options).expand(shape) * 0.1;
                }
                break;
            }
            case 4: {
                // Edge case: very small or large values
                float scale = (offset < size) ? (data[offset++] % 2 ? 1e-6 : 1e6) : 1.0;
                torch::Tensor B = torch::randn(shape, options) * scale;
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    A = torch::matmul(B, B.conj().transpose(-2, -1)) + torch::eye(n, options).expand(shape) * scale;
                } else {
                    A = torch::matmul(B, B.transpose(-2, -1)) + torch::eye(n, options).expand(shape) * scale;
                }
                break;
            }
            default:
                A = torch::eye(n, options).expand(shape);
        }
        
        // Test with output tensor
        bool use_out = (offset < size) ? data[offset++] % 2 : false;
        
        if (use_out) {
            // Create output tensor with same shape and dtype
            torch::Tensor out = torch::empty(shape, options);
            
            // Call cholesky with output tensor
            torch::linalg::cholesky_out(out, A, upper);
            
            // Verify result is in output tensor
            if (out.numel() > 0) {
                // Access some elements to ensure computation completed
                out.item<float>();  // This will throw if wrong dtype, which is fine
            }
        } else {
            // Call cholesky without output tensor
            torch::Tensor L = torch::linalg::cholesky(A, upper);
            
            // Basic validation - access result
            if (L.numel() > 0) {
                // Check that result has same shape as input
                if (L.sizes() != A.sizes()) {
                    std::cerr << "Shape mismatch in result" << std::endl;
                }
                
                // For upper=false, result should be lower triangular
                // For upper=true, result should be upper triangular
                // Just access some elements to ensure computation completed
                if (n > 1) {
                    if (upper) {
                        L.index({torch::indexing::Ellipsis, 1, 0});  // Should be zero for upper triangular
                    } else {
                        L.index({torch::indexing::Ellipsis, 0, 1});  // Should be zero for lower triangular
                    }
                }
            }
        }
        
        // Try edge cases if we have more data
        if (offset + 1 < size) {
            uint8_t edge_case = data[offset++] % 4;
            switch (edge_case) {
                case 0: {
                    // Empty tensor
                    torch::Tensor empty_A = torch::empty({0, 0}, options);
                    torch::Tensor empty_L = torch::linalg::cholesky(empty_A, upper);
                    break;
                }
                case 1: {
                    // 1x1 matrix
                    torch::Tensor tiny_A = torch::tensor({{2.0}}, options);
                    torch::Tensor tiny_L = torch::linalg::cholesky(tiny_A, upper);
                    break;
                }
                case 2: {
                    // Non-contiguous tensor
                    if (n > 1) {
                        torch::Tensor nc_A = A.transpose(-2, -1).contiguous().transpose(-2, -1);
                        torch::Tensor nc_L = torch::linalg::cholesky(nc_A, upper);
                    }
                    break;
                }
                case 3: {
                    // Different device (CPU only for fuzzing)
                    torch::Tensor cpu_A = A.cpu();
                    torch::Tensor cpu_L = torch::linalg::cholesky(cpu_A, upper);
                    break;
                }
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
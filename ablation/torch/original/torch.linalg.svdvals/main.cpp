#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 4) {
        // Need minimum bytes for configuration
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Parse configuration byte
        uint8_t config = data[offset++];
        bool use_out_tensor = (config & 0x01) != 0;
        bool use_cuda = (config & 0x02) != 0 && torch::cuda::is_available();
        bool use_batch = (config & 0x04) != 0;
        uint8_t driver_selector = (config >> 3) & 0x07; // 3 bits for driver selection
        
        // Parse dtype selector - only use supported dtypes for svdvals
        uint8_t dtype_selector = data[offset++];
        std::vector<torch::ScalarType> svd_dtypes = {
            torch::kFloat, torch::kDouble, 
            torch::kComplexFloat, torch::kComplexDouble
        };
        torch::ScalarType dtype = svd_dtypes[dtype_selector % svd_dtypes.size()];
        
        // Parse dimensions
        uint8_t batch_dims = 0;
        if (use_batch && offset < size) {
            batch_dims = data[offset++] % 3; // 0-2 batch dimensions
        }
        
        // Parse matrix dimensions m and n
        uint8_t m = 1, n = 1;
        if (offset < size) {
            m = (data[offset++] % 15) + 1; // 1-15 rows
        }
        if (offset < size) {
            n = (data[offset++] % 15) + 1; // 1-15 cols
        }
        
        // Build shape vector
        std::vector<int64_t> shape;
        if (use_batch) {
            for (uint8_t i = 0; i < batch_dims && offset < size; ++i) {
                uint8_t batch_size = (data[offset++] % 4) + 1; // 1-4 per batch dim
                shape.push_back(batch_size);
            }
        }
        shape.push_back(m);
        shape.push_back(n);
        
        // Calculate total elements
        int64_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }
        
        // Create input tensor A
        torch::Tensor A;
        auto options = torch::TensorOptions().dtype(dtype);
        
        if (offset < size && (size - offset) >= num_elements) {
            // Try to use fuzzer data for tensor values
            size_t dtype_size = c10::elementSize(dtype);
            std::vector<uint8_t> tensor_data(num_elements * dtype_size, 0);
            size_t bytes_to_copy = std::min(tensor_data.size(), size - offset);
            if (bytes_to_copy > 0) {
                std::memcpy(tensor_data.data(), data + offset, bytes_to_copy);
                offset += bytes_to_copy;
            }
            A = torch::from_blob(tensor_data.data(), shape, options).clone();
        } else {
            // Generate random tensor if not enough data
            A = torch::randn(shape, options);
        }
        
        // Move to CUDA if requested and available
        if (use_cuda) {
            A = A.cuda();
            options = options.device(torch::kCUDA);
        }
        
        // Prepare output tensor if requested
        torch::Tensor out;
        if (use_out_tensor) {
            // Output shape for svdvals is (*batch_dims, min(m, n))
            std::vector<int64_t> out_shape;
            if (use_batch) {
                for (uint8_t i = 0; i < batch_dims; ++i) {
                    out_shape.push_back(shape[i]);
                }
            }
            out_shape.push_back(std::min(m, n));
            
            // svdvals always returns real values even for complex input
            torch::ScalarType out_dtype = (dtype == torch::kComplexFloat) ? torch::kFloat :
                                         (dtype == torch::kComplexDouble) ? torch::kDouble : dtype;
            auto out_options = torch::TensorOptions().dtype(out_dtype);
            if (use_cuda) {
                out_options = out_options.device(torch::kCUDA);
            }
            out = torch::empty(out_shape, out_options);
        }
        
        // Select driver (only for CUDA)
        const char* driver = nullptr;
        if (use_cuda) {
            switch (driver_selector) {
                case 0: driver = nullptr; break;
                case 1: driver = "gesvd"; break;
                case 2: driver = "gesvdj"; break;
                case 3: driver = "gesvda"; break;
                default: driver = nullptr; break;
            }
        }
        
        // Call svdvals with different parameter combinations
        torch::Tensor result;
        
        if (use_out_tensor && driver) {
            // Both out and driver specified
            result = torch::linalg::svdvals_out(out, A, driver);
        } else if (use_out_tensor) {
            // Only out specified
            result = torch::linalg::svdvals_out(out, A);
        } else if (driver) {
            // Only driver specified (CUDA only)
            result = torch::linalg::svdvals(A, driver);
        } else {
            // No optional parameters
            result = torch::linalg::svdvals(A);
        }
        
        // Verify output properties
        if (result.defined()) {
            // Check that output is real-valued
            if (result.is_complex()) {
                std::cerr << "Error: svdvals should return real values" << std::endl;
                return -1;
            }
            
            // Check output shape
            std::vector<int64_t> expected_shape;
            if (use_batch) {
                for (uint8_t i = 0; i < batch_dims; ++i) {
                    expected_shape.push_back(shape[i]);
                }
            }
            expected_shape.push_back(std::min(static_cast<int64_t>(m), static_cast<int64_t>(n)));
            
            if (result.sizes().vec() != expected_shape) {
                std::cerr << "Shape mismatch: expected " << expected_shape.size() 
                         << " dims, got " << result.sizes().size() << std::endl;
            }
            
            // Check that singular values are in descending order
            if (result.numel() > 1) {
                auto result_cpu = result.cpu();
                auto result_flat = result_cpu.flatten();
                for (int64_t i = 0; i < result_flat.numel() - 1; ++i) {
                    // Only check within each matrix (not across batch dims)
                    int64_t singular_value_count = std::min(static_cast<int64_t>(m), static_cast<int64_t>(n));
                    if ((i + 1) % singular_value_count != 0) {
                        auto val1 = result_flat[i].item<double>();
                        auto val2 = result_flat[i + 1].item<double>();
                        if (val1 < val2 - 1e-6) { // Allow small numerical errors
                            std::cerr << "Warning: Singular values not in descending order at index " 
                                     << i << ": " << val1 << " < " << val2 << std::endl;
                        }
                    }
                }
            }
            
            // Test edge cases
            if (offset < size) {
                uint8_t edge_case = data[offset++];
                
                // Test with zero matrix
                if ((edge_case & 0x01) && shape.size() >= 2) {
                    auto zero_tensor = torch::zeros_like(A);
                    auto zero_result = torch::linalg::svdvals(zero_tensor);
                    // All singular values should be zero
                    if (torch::any(zero_result != 0).item<bool>()) {
                        std::cerr << "Warning: Zero matrix should have zero singular values" << std::endl;
                    }
                }
                
                // Test with identity-like matrix (if square)
                if ((edge_case & 0x02) && m == n) {
                    auto eye_tensor = torch::eye(m, options);
                    if (use_batch) {
                        // Expand to batch dimensions
                        std::vector<int64_t> expand_shape = shape;
                        eye_tensor = eye_tensor.expand(expand_shape);
                    }
                    if (use_cuda) {
                        eye_tensor = eye_tensor.cuda();
                    }
                    auto eye_result = torch::linalg::svdvals(eye_tensor);
                    // All singular values should be close to 1 for identity
                }
                
                // Test with NaN/Inf values (if floating point)
                if ((edge_case & 0x04) && !A.is_complex()) {
                    auto nan_tensor = A.clone();
                    nan_tensor[0] = std::numeric_limits<float>::quiet_NaN();
                    try {
                        auto nan_result = torch::linalg::svdvals(nan_tensor);
                        // Check if NaN propagated
                        if (torch::any(torch::isnan(nan_result)).item<bool>()) {
                            // Expected behavior - NaN propagates
                        }
                    } catch (const std::exception& e) {
                        // Some implementations might throw on NaN
                    }
                }
            }
        }
        
        // Additional consistency check: compare with full SVD if small enough
        if (num_elements < 1000 && !use_cuda) { // Only for small CPU tensors
            try {
                auto [U, S, Vh] = torch::linalg::svd(A, false);
                if (!torch::allclose(result, S, 1e-5, 1e-8)) {
                    std::cerr << "Warning: svdvals result differs from svd().S" << std::endl;
                }
            } catch (const std::exception& e) {
                // SVD might fail where svdvals succeeds, that's ok
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
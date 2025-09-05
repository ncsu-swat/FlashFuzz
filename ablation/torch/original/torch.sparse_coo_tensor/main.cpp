#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 10) {
        // Need minimum bytes for basic parameters
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Parse control flags from first few bytes
        uint8_t sparse_dims = (data[offset++] % 4) + 1;  // 1-4 sparse dimensions
        uint8_t dense_dims = data[offset++] % 3;  // 0-2 dense dimensions
        uint8_t nnz = data[offset++] % 32;  // 0-31 non-zero elements
        bool provide_size = data[offset++] & 1;
        bool provide_dtype = data[offset++] & 1;
        bool requires_grad = data[offset++] & 1;
        bool check_invariants = data[offset++] & 1;
        uint8_t coalesce_flag = data[offset++] % 3;  // 0=None, 1=False, 2=True
        uint8_t device_type = data[offset++] % 2;  // 0=CPU, 1=CUDA if available
        
        // Parse dtype for values if provided
        torch::ScalarType values_dtype = torch::kFloat32;
        if (provide_dtype && offset < size) {
            values_dtype = fuzzer_utils::parseDataType(data[offset++]);
        }
        
        // Create indices tensor (shape: [sparse_dims, nnz])
        std::vector<int64_t> indices_shape = {sparse_dims, nnz};
        std::vector<int64_t> indices_data;
        
        // Generate indices data - need to be valid coordinates
        std::vector<int64_t> max_dims(sparse_dims, 1);
        if (offset + sparse_dims * sizeof(uint8_t) <= size) {
            for (int i = 0; i < sparse_dims; i++) {
                max_dims[i] = (data[offset++] % 15) + 1;  // Each dimension 1-15
            }
        }
        
        // Fill indices with valid coordinates
        for (int64_t i = 0; i < nnz; i++) {
            for (int64_t d = 0; d < sparse_dims; d++) {
                if (offset < size) {
                    int64_t coord = data[offset++] % max_dims[d];
                    indices_data.push_back(coord);
                } else {
                    indices_data.push_back(0);
                }
            }
        }
        
        // Create indices tensor
        torch::Tensor indices;
        if (nnz > 0) {
            indices = torch::from_blob(indices_data.data(), indices_shape, torch::kInt64).clone();
        } else {
            indices = torch::empty({sparse_dims, 0}, torch::kInt64);
        }
        
        // Create values tensor
        torch::Tensor values;
        if (dense_dims > 0) {
            // Values shape: [nnz, dense_dim1, dense_dim2, ...]
            std::vector<int64_t> values_shape;
            values_shape.push_back(nnz);
            for (int i = 0; i < dense_dims; i++) {
                if (offset < size) {
                    int64_t dim = (data[offset++] % 4) + 1;  // 1-4 for dense dims
                    values_shape.push_back(dim);
                } else {
                    values_shape.push_back(1);
                }
            }
            
            // Create values tensor with fuzzer data or random
            if (offset + 4 <= size) {
                values = fuzzer_utils::createTensor(data, size, offset);
                // Reshape to match expected shape
                int64_t total_elements = 1;
                for (auto dim : values_shape) {
                    total_elements *= dim;
                }
                if (values.numel() >= total_elements) {
                    values = values.narrow(0, 0, total_elements).reshape(values_shape);
                } else {
                    values = torch::randn(values_shape, torch::dtype(values_dtype));
                }
                values = values.to(values_dtype);
            } else {
                values = torch::randn(values_shape, torch::dtype(values_dtype));
            }
        } else {
            // Simple 1D values tensor for sparse-only tensor
            if (nnz > 0) {
                if (offset + 4 <= size) {
                    values = fuzzer_utils::createTensor(data, size, offset);
                    if (values.numel() >= nnz) {
                        values = values.narrow(0, 0, nnz);
                    } else {
                        values = torch::randn({nnz}, torch::dtype(values_dtype));
                    }
                    values = values.to(values_dtype);
                } else {
                    values = torch::randn({nnz}, torch::dtype(values_dtype));
                }
            } else {
                values = torch::empty({0}, torch::dtype(values_dtype));
            }
        }
        
        // Prepare size argument if needed
        std::vector<int64_t> tensor_size;
        if (provide_size) {
            // Ensure size is at least as large as max indices
            for (int i = 0; i < sparse_dims; i++) {
                tensor_size.push_back(max_dims[i]);
            }
            // Add dense dimensions if any
            for (int i = 0; i < dense_dims; i++) {
                if (offset < size) {
                    tensor_size.push_back((data[offset++] % 4) + 1);
                } else {
                    tensor_size.push_back(2);
                }
            }
        }
        
        // Set device
        torch::Device device(torch::kCPU);
#ifdef USE_GPU
        if (device_type == 1 && torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            indices = indices.to(device);
            values = values.to(device);
        }
#endif
        
        // Create sparse COO tensor with various parameter combinations
        torch::Tensor sparse_tensor;
        
        if (!provide_size) {
            // Test without explicit size
            if (coalesce_flag == 0) {
                // is_coalesced = None (default)
                sparse_tensor = torch::sparse_coo_tensor(
                    indices, values,
                    torch::TensorOptions()
                        .dtype(provide_dtype ? values_dtype : values.dtype())
                        .device(device)
                        .requires_grad(requires_grad && values.dtype().isFloatingPoint())
                );
            } else {
                // is_coalesced = true/false
                bool is_coalesced = (coalesce_flag == 2);
                sparse_tensor = torch::sparse_coo_tensor(
                    indices, values,
                    torch::TensorOptions()
                        .dtype(provide_dtype ? values_dtype : values.dtype())
                        .device(device)
                        .requires_grad(requires_grad && values.dtype().isFloatingPoint())
                );
                
                // Note: C++ API doesn't have direct is_coalesced parameter,
                // but we can test coalesce operation
                if (is_coalesced) {
                    sparse_tensor = sparse_tensor.coalesce();
                }
            }
        } else {
            // Test with explicit size
            sparse_tensor = torch::sparse_coo_tensor(
                indices, values, tensor_size,
                torch::TensorOptions()
                    .dtype(provide_dtype ? values_dtype : values.dtype())
                    .device(device)
                    .requires_grad(requires_grad && values.dtype().isFloatingPoint())
            );
            
            if (coalesce_flag == 2) {
                sparse_tensor = sparse_tensor.coalesce();
            }
        }
        
        // Perform operations to exercise the sparse tensor
        if (sparse_tensor.numel() > 0) {
            // Test basic properties
            auto nnz_result = sparse_tensor._nnz();
            auto indices_result = sparse_tensor._indices();
            auto values_result = sparse_tensor._values();
            auto sparse_dim = sparse_tensor.sparse_dim();
            auto dense_dim = sparse_tensor.dense_dim();
            
            // Test coalesce if not already done
            if (coalesce_flag != 2 && nnz > 1) {
                auto coalesced = sparse_tensor.coalesce();
                bool is_coalesced = coalesced.is_coalesced();
                (void)is_coalesced; // Suppress unused warning
            }
            
            // Test conversion to dense if tensor is small enough
            if (sparse_tensor.numel() < 1000) {
                auto dense = sparse_tensor.to_dense();
                
                // Test round-trip conversion
                auto sparse_again = dense.to_sparse();
                
                // For floating point types, test some operations
                if (sparse_tensor.dtype().isFloatingPoint()) {
                    // Test addition
                    auto sum_result = sparse_tensor + sparse_tensor;
                    
                    // Test scalar multiplication
                    auto mul_result = sparse_tensor * 2.0;
                    
                    // Test transpose if 2D
                    if (sparse_tensor.dim() == 2) {
                        auto transposed = sparse_tensor.t();
                    }
                }
            }
            
            // Test cloning
            auto cloned = sparse_tensor.clone();
            
            // Test type conversion
            if (sparse_tensor.dtype() != torch::kFloat64) {
                auto converted = sparse_tensor.to(torch::kFloat64);
            }
        }
        
        // Test edge cases with empty tensors
        if (offset % 7 == 0) {
            // Create empty sparse tensor with various configurations
            auto empty_indices = torch::empty({2, 0}, torch::kInt64);
            auto empty_values = torch::empty({0}, torch::kFloat32);
            auto empty_sparse = torch::sparse_coo_tensor(empty_indices, empty_values, {10, 10});
            
            // Test operations on empty sparse tensor
            auto nnz = empty_sparse._nnz();
            auto dense = empty_sparse.to_dense();
        }
        
    } catch (const c10::Error &e) {
        // PyTorch-specific errors are expected during fuzzing
#ifdef DEBUG_FUZZ
        std::cout << "PyTorch error: " << e.what() << std::endl;
#endif
        return 0;  // Continue fuzzing
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected errors
    }
    
    return 0;
}
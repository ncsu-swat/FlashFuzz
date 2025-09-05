#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need minimum bytes for basic parsing
        if (Size < 10) {
            return 0;
        }

        // Parse configuration bytes
        uint8_t sparse_dim = Data[offset++] % 5 + 1;  // 1-5 sparse dimensions
        uint8_t dense_dim = Data[offset++] % 3;       // 0-2 dense dimensions
        uint8_t nnz = Data[offset++] % 32 + 1;        // 1-32 non-zero elements
        bool requires_grad = Data[offset++] & 1;
        bool is_coalesced = Data[offset++] & 1;
        
        // Parse shape for the sparse tensor
        std::vector<int64_t> shape;
        uint8_t total_dims = sparse_dim + dense_dim;
        for (uint8_t i = 0; i < total_dims; ++i) {
            if (offset >= Size) break;
            int64_t dim_size = (Data[offset++] % 10) + 1;  // 1-10 per dimension
            shape.push_back(dim_size);
        }
        
        // Ensure we have enough dimensions
        while (shape.size() < total_dims) {
            shape.push_back(2);  // Default size
        }
        
        // Create indices tensor - shape should be [sparse_dim, nnz]
        std::vector<int64_t> indices_shape = {sparse_dim, nnz};
        torch::Tensor indices;
        
        if (offset < Size && Data[offset++] & 1) {
            // Use fuzzer data for indices
            std::vector<int64_t> indices_data;
            for (int64_t i = 0; i < sparse_dim * nnz; ++i) {
                if (offset >= Size) {
                    indices_data.push_back(0);
                } else {
                    // Ensure indices are within bounds for each dimension
                    int64_t dim_idx = i / nnz;  // Which dimension
                    int64_t max_val = (dim_idx < shape.size()) ? shape[dim_idx] : 1;
                    indices_data.push_back(Data[offset++] % max_val);
                }
            }
            indices = torch::from_blob(indices_data.data(), indices_shape, 
                                      torch::TensorOptions().dtype(torch::kLong)).clone();
        } else {
            // Generate random valid indices
            indices = torch::zeros(indices_shape, torch::TensorOptions().dtype(torch::kLong));
            for (int64_t i = 0; i < sparse_dim; ++i) {
                if (i < shape.size()) {
                    indices[i] = torch::randint(0, shape[i], {nnz});
                }
            }
        }
        
        // Determine values shape based on dense dimensions
        std::vector<int64_t> values_shape = {nnz};
        for (int64_t i = sparse_dim; i < shape.size(); ++i) {
            values_shape.push_back(shape[i]);
        }
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            // Try to create values tensor from fuzzer data
            try {
                values = fuzzer_utils::createTensor(Data, Size, offset);
                // Reshape to match expected values shape
                int64_t total_elements = 1;
                for (auto dim : values_shape) {
                    total_elements *= dim;
                }
                if (values.numel() >= total_elements) {
                    values = values.flatten().slice(0, 0, total_elements).reshape(values_shape);
                } else {
                    // Pad with zeros if needed
                    values = torch::zeros(values_shape, values.options());
                }
            } catch (...) {
                // Fallback to random values
                values = torch::randn(values_shape);
            }
        } else {
            values = torch::randn(values_shape);
        }
        
        // Test 1: Basic sparse_coo_tensor creation
        torch::Tensor sparse_tensor;
        try {
            sparse_tensor = torch::sparse_coo_tensor(indices, values, shape,
                torch::TensorOptions().requires_grad(requires_grad));
        } catch (const c10::Error& e) {
            // Some combinations might be invalid, that's ok for fuzzing
            return 0;
        }
        
        // Test 2: Try with different dtypes
        if (offset < Size) {
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            try {
                values = values.to(dtype);
                auto sparse_tensor2 = torch::sparse_coo_tensor(indices, values, shape,
                    torch::TensorOptions().dtype(dtype));
                
                // Test operations on sparse tensor
                if (sparse_tensor2.numel() > 0) {
                    auto sum_result = sparse_tensor2.sum();
                    auto mean_result = sparse_tensor2.mean();
                }
            } catch (...) {
                // Some dtypes might not be supported for sparse, continue
            }
        }
        
        // Test 3: Coalesce operation
        if (is_coalesced && sparse_tensor.defined()) {
            try {
                auto coalesced = sparse_tensor.coalesce();
                // Verify coalesced tensor
                if (coalesced.is_coalesced()) {
                    auto indices_coalesced = coalesced.indices();
                    auto values_coalesced = coalesced.values();
                }
            } catch (...) {
                // Coalesce might fail for some inputs
            }
        }
        
        // Test 4: Convert to dense and back
        if (sparse_tensor.defined() && sparse_tensor.numel() < 1000) {
            try {
                auto dense = sparse_tensor.to_dense();
                auto sparse_again = dense.to_sparse();
                
                // Compare if conversion preserves values
                if (dense.numel() > 0) {
                    auto diff = (sparse_again.to_dense() - dense).abs().max();
                }
            } catch (...) {
                // Conversion might fail for some configurations
            }
        }
        
        // Test 5: Edge cases with empty tensors
        if (offset < Size && Data[offset++] & 1) {
            try {
                // Empty indices
                auto empty_indices = torch::zeros({sparse_dim, 0}, torch::kLong);
                auto empty_values = torch::zeros({0});
                auto empty_sparse = torch::sparse_coo_tensor(empty_indices, empty_values, shape);
            } catch (...) {
                // Expected to potentially fail
            }
        }
        
        // Test 6: Test with size inference
        if (offset < Size && Data[offset++] & 1) {
            try {
                // Let PyTorch infer the size
                auto inferred_sparse = torch::sparse_coo_tensor(indices, values);
                if (inferred_sparse.defined()) {
                    auto inferred_shape = inferred_sparse.sizes();
                }
            } catch (...) {
                // Size inference might fail
            }
        }
        
        // Test 7: Test with different layouts and devices
        if (sparse_tensor.defined()) {
            try {
                // Try operations that might trigger different code paths
                auto t = sparse_tensor.t();  // Transpose
                auto neg = -sparse_tensor;   // Negation
                
                if (sparse_tensor.dim() >= 2) {
                    auto reshaped = sparse_tensor.reshape({-1});
                }
            } catch (...) {
                // Some operations might not be supported
            }
        }
        
        // Test 8: Hybrid sparse tensors (with dense dimensions)
        if (dense_dim > 0 && offset < Size) {
            try {
                // Create a hybrid sparse tensor
                std::vector<int64_t> hybrid_values_shape = {nnz};
                for (int i = 0; i < dense_dim; ++i) {
                    hybrid_values_shape.push_back(shape[sparse_dim + i]);
                }
                auto hybrid_values = torch::randn(hybrid_values_shape);
                auto hybrid_sparse = torch::sparse_coo_tensor(
                    indices.slice(0, 0, sparse_dim),  // Only sparse dimensions in indices
                    hybrid_values,
                    shape
                );
            } catch (...) {
                // Hybrid tensors might have specific requirements
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic parameters
    }

    try {
        size_t offset = 0;
        
        // Helper lambda to consume bytes
        auto consumeBytes = [&](size_t num_bytes) -> std::vector<uint8_t> {
            if (offset + num_bytes > size) {
                return std::vector<uint8_t>(num_bytes, 0);
            }
            std::vector<uint8_t> result(data + offset, data + offset + num_bytes);
            offset += num_bytes;
            return result;
        };
        
        auto consumeUInt8 = [&]() -> uint8_t {
            if (offset >= size) return 0;
            return data[offset++];
        };
        
        auto consumeInt64 = [&]() -> int64_t {
            if (offset + 8 > size) return 1;
            int64_t val = 0;
            std::memcpy(&val, data + offset, 8);
            offset += 8;
            return std::abs(val) % 100 + 1;  // Limit size for memory safety
        };
        
        auto consumeFloat = [&]() -> float {
            if (offset + 4 > size) return 0.0f;
            float val = 0;
            std::memcpy(&val, data + offset, 4);
            offset += 4;
            return val;
        };

        // Determine tensor dimensions
        uint8_t sparse_dim = (consumeUInt8() % 4) + 1;  // 1-4 dimensions
        uint8_t dense_dim = consumeUInt8() % 3;  // 0-2 dense dimensions
        int64_t nnz = consumeInt64() % 50;  // Number of non-zero elements
        
        // Determine dtype
        uint8_t dtype_choice = consumeUInt8() % 5;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Determine device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available() && (consumeUInt8() % 4 == 0)) {
            device = torch::kCUDA;
        }
        
        // Build indices tensor (sparse_dim x nnz)
        std::vector<int64_t> indices_data;
        std::vector<int64_t> max_indices(sparse_dim, 1);
        
        for (int64_t i = 0; i < sparse_dim; ++i) {
            max_indices[i] = (consumeInt64() % 20) + 1;  // Size in each sparse dimension
        }
        
        for (int64_t i = 0; i < sparse_dim * nnz; ++i) {
            int64_t dim_idx = i / nnz;
            int64_t idx = consumeInt64() % max_indices[dim_idx];
            indices_data.push_back(idx);
        }
        
        auto indices = torch::from_blob(indices_data.data(), 
                                       {sparse_dim, nnz}, 
                                       torch::kInt64).clone();
        
        // Build values tensor
        std::vector<int64_t> values_shape;
        values_shape.push_back(nnz);
        for (int64_t i = 0; i < dense_dim; ++i) {
            values_shape.push_back((consumeInt64() % 10) + 1);
        }
        
        int64_t total_values = 1;
        for (auto s : values_shape) {
            total_values *= s;
        }
        
        torch::Tensor values;
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            std::vector<float> values_data;
            for (int64_t i = 0; i < total_values; ++i) {
                values_data.push_back(consumeFloat());
            }
            values = torch::from_blob(values_data.data(), values_shape, torch::kFloat32).clone();
            if (dtype == torch::kFloat64) {
                values = values.to(torch::kFloat64);
            }
        } else {
            std::vector<int64_t> values_data;
            for (int64_t i = 0; i < total_values; ++i) {
                values_data.push_back(consumeInt64());
            }
            values = torch::from_blob(values_data.data(), values_shape, torch::kInt64).clone();
            if (dtype == torch::kInt32) {
                values = values.to(torch::kInt32);
            }
        }
        
        // Build size vector
        std::vector<int64_t> tensor_size;
        bool use_explicit_size = consumeUInt8() % 2;
        
        if (use_explicit_size) {
            // Explicit size
            for (int64_t i = 0; i < sparse_dim; ++i) {
                tensor_size.push_back(max_indices[i] + (consumeInt64() % 5));
            }
            for (int64_t i = 0; i < dense_dim; ++i) {
                tensor_size.push_back(values_shape[i + 1]);
            }
        }
        
        // Determine optional parameters
        bool requires_grad = (consumeUInt8() % 2) && (dtype == torch::kFloat32 || dtype == torch::kFloat64);
        
        uint8_t check_invariants_choice = consumeUInt8() % 3;
        c10::optional<bool> check_invariants;
        if (check_invariants_choice == 1) check_invariants = true;
        else if (check_invariants_choice == 2) check_invariants = false;
        
        uint8_t is_coalesced_choice = consumeUInt8() % 3;
        c10::optional<bool> is_coalesced;
        if (is_coalesced_choice == 1) is_coalesced = true;
        else if (is_coalesced_choice == 2) is_coalesced = false;
        
        // Move tensors to device
        indices = indices.to(device);
        values = values.to(device);
        
        // Create sparse COO tensor
        torch::Tensor sparse_tensor;
        
        if (use_explicit_size) {
            sparse_tensor = torch::sparse_coo_tensor(
                indices,
                values,
                tensor_size,
                torch::TensorOptions()
                    .dtype(dtype)
                    .device(device)
                    .requires_grad(requires_grad)
            );
        } else {
            // Size inference
            sparse_tensor = torch::sparse_coo_tensor(
                indices,
                values,
                c10::nullopt,
                torch::TensorOptions()
                    .dtype(dtype)
                    .device(device)
                    .requires_grad(requires_grad)
            );
        }
        
        // Perform some operations to exercise the tensor
        if (sparse_tensor.numel() > 0) {
            // Test coalesce
            auto coalesced = sparse_tensor.coalesce();
            
            // Test to_dense if reasonable size
            if (sparse_tensor.numel() < 10000) {
                auto dense = sparse_tensor.to_dense();
                
                // Test basic operations
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    auto sum = sparse_tensor.sum();
                    auto mean = sparse_tensor.mean();
                }
            }
            
            // Test sparse_dim and dense_dim
            auto sdim = sparse_tensor.sparse_dim();
            auto ddim = sparse_tensor.dense_dim();
            
            // Test indices and values access
            auto idx = sparse_tensor._indices();
            auto vals = sparse_tensor._values();
            
            // Test is_coalesced check
            bool coalesced_status = sparse_tensor.is_coalesced();
        }
        
        // Test edge cases with empty tensors
        if (consumeUInt8() % 10 == 0) {
            auto empty_indices = torch::empty({2, 0}, torch::kInt64);
            auto empty_values = torch::empty({0}, dtype);
            auto empty_sparse = torch::sparse_coo_tensor(
                empty_indices,
                empty_values,
                {10, 10},
                torch::TensorOptions().dtype(dtype)
            );
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 16) return 0; // Need minimum bytes for basic tensor creation
        
        size_t offset = 0;
        
        // Consume configuration bytes
        uint8_t num_dims1, num_dims2, dtype_idx1, dtype_idx2;
        uint8_t use_same_shape, use_scalar1, use_scalar2, use_empty;
        
        if (!consumeBytes(Data, offset, Size, num_dims1)) return 0;
        if (!consumeBytes(Data, offset, Size, num_dims2)) return 0;
        if (!consumeBytes(Data, offset, Size, dtype_idx1)) return 0;
        if (!consumeBytes(Data, offset, Size, dtype_idx2)) return 0;
        if (!consumeBytes(Data, offset, Size, use_same_shape)) return 0;
        if (!consumeBytes(Data, offset, Size, use_scalar1)) return 0;
        if (!consumeBytes(Data, offset, Size, use_scalar2)) return 0;
        if (!consumeBytes(Data, offset, Size, use_empty)) return 0;
        
        // Map to reasonable ranges
        num_dims1 = num_dims1 % 5; // 0-4 dimensions
        num_dims2 = num_dims2 % 5;
        
        // Select dtypes (only integer types make sense for lcm)
        std::vector<torch::ScalarType> int_dtypes = {
            torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong
        };
        torch::ScalarType dtype1 = int_dtypes[dtype_idx1 % int_dtypes.size()];
        torch::ScalarType dtype2 = int_dtypes[dtype_idx2 % int_dtypes.size()];
        
        // Create first tensor
        torch::Tensor tensor1;
        if (use_scalar1 & 1) {
            // Create scalar tensor
            int64_t scalar_val;
            if (!consumeBytes(Data, offset, Size, scalar_val)) scalar_val = 1;
            tensor1 = torch::tensor(scalar_val, torch::dtype(dtype1));
        } else if (use_empty & 1) {
            // Create empty tensor
            tensor1 = torch::empty({0}, torch::dtype(dtype1));
        } else {
            // Create tensor with shape
            std::vector<int64_t> shape1;
            for (uint8_t i = 0; i < num_dims1; ++i) {
                uint8_t dim_size;
                if (!consumeBytes(Data, offset, Size, dim_size)) dim_size = 1;
                shape1.push_back((dim_size % 10) + (use_empty & 2 ? 0 : 1)); // 0-10 or 1-10
            }
            if (shape1.empty()) shape1.push_back(1);
            
            // Calculate total elements
            int64_t total_elems = 1;
            for (auto d : shape1) total_elems *= d;
            
            // Limit total elements to prevent OOM
            if (total_elems > 10000) {
                for (auto& d : shape1) d = std::min(d, (int64_t)10);
                total_elems = 1;
                for (auto d : shape1) total_elems *= d;
            }
            
            // Fill with data from fuzzer input
            std::vector<int64_t> data_vec;
            for (int64_t i = 0; i < total_elems; ++i) {
                int64_t val;
                if (!consumeBytes(Data, offset, Size, val)) {
                    val = i + 1; // Default non-zero values
                }
                data_vec.push_back(val);
            }
            
            tensor1 = torch::from_blob(data_vec.data(), shape1, torch::kLong).to(dtype1).clone();
        }
        
        // Create second tensor
        torch::Tensor tensor2;
        if (use_same_shape & 1) {
            // Use same shape as tensor1
            if (tensor1.numel() > 0) {
                std::vector<int64_t> data_vec;
                for (int64_t i = 0; i < tensor1.numel(); ++i) {
                    int64_t val;
                    if (!consumeBytes(Data, offset, Size, val)) {
                        val = (i + 2) % 100; // Different default values
                    }
                    data_vec.push_back(val);
                }
                tensor2 = torch::from_blob(data_vec.data(), tensor1.sizes(), torch::kLong).to(dtype2).clone();
            } else {
                tensor2 = torch::empty_like(tensor1).to(dtype2);
            }
        } else if (use_scalar2 & 1) {
            // Create scalar tensor
            int64_t scalar_val;
            if (!consumeBytes(Data, offset, Size, scalar_val)) scalar_val = 2;
            tensor2 = torch::tensor(scalar_val, torch::dtype(dtype2));
        } else {
            // Create tensor with different shape
            std::vector<int64_t> shape2;
            for (uint8_t i = 0; i < num_dims2; ++i) {
                uint8_t dim_size;
                if (!consumeBytes(Data, offset, Size, dim_size)) dim_size = 1;
                shape2.push_back((dim_size % 10) + 1); // 1-10
            }
            if (shape2.empty()) shape2.push_back(1);
            
            // Calculate total elements
            int64_t total_elems = 1;
            for (auto d : shape2) total_elems *= d;
            
            // Limit total elements
            if (total_elems > 10000) {
                for (auto& d : shape2) d = std::min(d, (int64_t)10);
                total_elems = 1;
                for (auto d : shape2) total_elems *= d;
            }
            
            // Fill with data
            std::vector<int64_t> data_vec;
            for (int64_t i = 0; i < total_elems; ++i) {
                int64_t val;
                if (!consumeBytes(Data, offset, Size, val)) {
                    val = (i + 3) % 100;
                }
                data_vec.push_back(val);
            }
            
            tensor2 = torch::from_blob(data_vec.data(), shape2, torch::kLong).to(dtype2).clone();
        }
        
        // Apply lcm_ operation (in-place)
        try {
            tensor1.lcm_(tensor2);
            
            // Verify result is valid (no inf/nan for integer types)
            if (tensor1.numel() > 0) {
                auto flat = tensor1.flatten();
                // Access some elements to ensure computation completed
                if (flat.numel() > 0) {
                    auto first = flat[0].item<int64_t>();
                    (void)first; // Suppress unused warning
                }
            }
        } catch (const c10::Error& e) {
            // Expected errors for shape mismatches, etc.
            return 0;
        } catch (const std::runtime_error& e) {
            // Expected runtime errors
            return 0;
        }
        
        // Test edge cases with special values
        uint8_t test_special;
        if (consumeBytes(Data, offset, Size, test_special)) {
            if (test_special & 1) {
                // Test with zeros
                auto zero_tensor = torch::zeros({2, 2}, torch::kInt);
                auto ones_tensor = torch::ones({2, 2}, torch::kInt);
                try {
                    zero_tensor.lcm_(ones_tensor);
                } catch (...) {
                    // Ignore errors for edge cases
                }
            }
            if (test_special & 2) {
                // Test with negative values
                auto neg_tensor = torch::tensor({{-1, -2}, {-3, -4}}, torch::kInt);
                auto pos_tensor = torch::tensor({{1, 2}, {3, 4}}, torch::kInt);
                try {
                    neg_tensor.lcm_(pos_tensor);
                } catch (...) {
                    // Ignore errors
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
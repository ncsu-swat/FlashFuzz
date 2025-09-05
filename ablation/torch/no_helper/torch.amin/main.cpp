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
        
        // Extract basic parameters from fuzzer input
        uint8_t rank = data[offset++] % 5 + 1;  // Tensor rank 1-5
        uint8_t dtype_selector = data[offset++] % 4;  // Select dtype
        bool keepdim = data[offset++] & 1;
        uint8_t dim_count = data[offset++] % rank + 1;  // Number of dims to reduce
        
        // Build tensor shape
        std::vector<int64_t> shape;
        for (size_t i = 0; i < rank && offset < size; ++i) {
            int64_t dim_size = (data[offset++] % 10) + 1;  // Dimension size 1-10
            shape.push_back(dim_size);
        }
        
        // If we ran out of data, use default shape
        while (shape.size() < rank) {
            shape.push_back(2);
        }
        
        // Build dimensions to reduce
        std::vector<int64_t> dims;
        std::vector<bool> used_dims(rank, false);
        for (size_t i = 0; i < dim_count && offset < size; ++i) {
            int64_t dim = data[offset++] % rank;
            if (!used_dims[dim]) {
                dims.push_back(dim);
                used_dims[dim] = true;
            }
        }
        
        // Ensure we have at least one dimension
        if (dims.empty()) {
            dims.push_back(0);
        }
        
        // Create tensor with appropriate dtype
        torch::Tensor input;
        torch::TensorOptions options;
        
        switch (dtype_selector) {
            case 0:
                options = torch::TensorOptions().dtype(torch::kFloat32);
                break;
            case 1:
                options = torch::TensorOptions().dtype(torch::kFloat64);
                break;
            case 2:
                options = torch::TensorOptions().dtype(torch::kInt32);
                break;
            case 3:
                options = torch::TensorOptions().dtype(torch::kInt64);
                break;
        }
        
        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }
        
        // Fill tensor with fuzzer data
        if (dtype_selector <= 1) {  // Floating point types
            std::vector<float> values;
            for (int64_t i = 0; i < total_elements && offset < size; ++i) {
                // Create various float values including special cases
                uint8_t val_type = data[offset++];
                float val;
                if (val_type < 200) {
                    val = static_cast<float>(val_type - 100) / 10.0f;
                } else if (val_type < 210) {
                    val = std::numeric_limits<float>::infinity();
                } else if (val_type < 220) {
                    val = -std::numeric_limits<float>::infinity();
                } else if (val_type < 230) {
                    val = std::numeric_limits<float>::quiet_NaN();
                } else {
                    val = 0.0f;
                }
                values.push_back(val);
            }
            
            // Pad with zeros if needed
            while (values.size() < total_elements) {
                values.push_back(0.0f);
            }
            
            input = torch::from_blob(values.data(), shape, torch::kFloat32).clone();
            if (dtype_selector == 1) {
                input = input.to(torch::kFloat64);
            }
        } else {  // Integer types
            std::vector<int32_t> values;
            for (int64_t i = 0; i < total_elements && offset < size; ++i) {
                int32_t val = static_cast<int32_t>(data[offset++]) - 128;
                values.push_back(val);
            }
            
            // Pad with zeros if needed
            while (values.size() < total_elements) {
                values.push_back(0);
            }
            
            input = torch::from_blob(values.data(), shape, torch::kInt32).clone();
            if (dtype_selector == 3) {
                input = input.to(torch::kInt64);
            }
        }
        
        // Test with single dimension
        if (offset < size && (data[offset++] & 1)) {
            int64_t single_dim = dims[0];
            torch::Tensor result1 = torch::amin(input, single_dim, keepdim);
            
            // Also test with optional out tensor
            if (offset < size && (data[offset++] & 1)) {
                auto out_shape = input.sizes().vec();
                if (keepdim) {
                    out_shape[single_dim] = 1;
                } else {
                    out_shape.erase(out_shape.begin() + single_dim);
                }
                torch::Tensor out = torch::empty(out_shape, options);
                torch::amin_out(out, input, single_dim, keepdim);
            }
        }
        
        // Test with multiple dimensions
        torch::Tensor result2 = torch::amin(input, dims, keepdim);
        
        // Test edge cases
        if (offset < size && (data[offset++] & 1)) {
            // Test with negative dimensions
            std::vector<int64_t> neg_dims;
            for (auto d : dims) {
                neg_dims.push_back(d - rank);
            }
            torch::Tensor result3 = torch::amin(input, neg_dims, keepdim);
        }
        
        // Test with empty tensor
        if (offset < size && (data[offset++] & 1)) {
            torch::Tensor empty_tensor = torch::empty({0, 3, 4}, options);
            if (!empty_tensor.numel()) {
                torch::Tensor empty_result = torch::amin(empty_tensor, {1}, keepdim);
            }
        }
        
        // Test with scalar tensor
        if (offset < size && (data[offset++] & 1)) {
            torch::Tensor scalar = torch::tensor(3.14f);
            torch::Tensor scalar_result = torch::amin(scalar, {}, keepdim);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
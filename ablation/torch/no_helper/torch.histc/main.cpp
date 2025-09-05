#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 16) {
        return 0; // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;
        
        // Extract basic parameters
        uint8_t tensor_rank = data[offset++] % 5; // 0-4 dimensions
        uint8_t dtype_selector = data[offset++] % 4; // Select dtype
        uint32_t bins = 1 + (data[offset] | (data[offset+1] << 8)); // 1-65536 bins
        offset += 2;
        
        // Extract min/max values
        float min_val, max_val;
        if (offset + 8 <= size) {
            memcpy(&min_val, data + offset, 4);
            memcpy(&max_val, data + offset + 4, 4);
            offset += 8;
        } else {
            min_val = 0.0f;
            max_val = 0.0f;
        }
        
        // Handle NaN/Inf cases
        if (std::isnan(min_val)) min_val = 0.0f;
        if (std::isnan(max_val)) max_val = 0.0f;
        if (std::isinf(min_val)) min_val = -1e6f;
        if (std::isinf(max_val)) max_val = 1e6f;
        
        // Build tensor shape
        std::vector<int64_t> shape;
        if (tensor_rank == 0) {
            shape = {}; // Scalar tensor
        } else {
            for (uint8_t i = 0; i < tensor_rank && offset < size; i++) {
                int64_t dim = (data[offset++] % 16) + 1; // 1-16 per dimension
                shape.push_back(dim);
            }
        }
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 10000) {
            total_elements = 10000;
            if (!shape.empty()) {
                shape[0] = total_elements / std::max(int64_t(1), 
                    std::accumulate(shape.begin() + 1, shape.end(), 
                                   int64_t(1), std::multiplies<int64_t>()));
            }
        }
        
        // Create input tensor with various dtypes
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
        
        if (shape.empty()) {
            // Scalar tensor
            if (offset < size) {
                float scalar_val;
                if (offset + 4 <= size) {
                    memcpy(&scalar_val, data + offset, 4);
                } else {
                    scalar_val = static_cast<float>(data[offset]);
                }
                input = torch::tensor(scalar_val, options);
            } else {
                input = torch::tensor(1.0f, options);
            }
        } else {
            // Multi-dimensional tensor
            input = torch::empty(shape, options);
            
            // Fill tensor with fuzzed data
            if (dtype_selector <= 1) { // Float types
                auto input_data = input.data_ptr<float>();
                for (int64_t i = 0; i < total_elements && offset < size; i++) {
                    if (offset + 4 <= size) {
                        float val;
                        memcpy(&val, data + offset, 4);
                        input_data[i] = val;
                        offset += 4;
                    } else {
                        input_data[i] = static_cast<float>(data[offset++]);
                    }
                }
            } else { // Integer types
                auto input_data = input.data_ptr<int32_t>();
                for (int64_t i = 0; i < total_elements && offset < size; i++) {
                    if (offset + 4 <= size) {
                        int32_t val;
                        memcpy(&val, data + offset, 4);
                        input_data[i] = val;
                        offset += 4;
                    } else {
                        input_data[i] = static_cast<int32_t>(data[offset++]);
                    }
                }
            }
        }
        
        // Test different scenarios
        uint8_t test_case = (offset < size) ? data[offset++] % 6 : 0;
        
        torch::Tensor result;
        switch (test_case) {
            case 0:
                // Basic histc with specified bins, min, max
                result = torch::histc(input, bins, min_val, max_val);
                break;
            case 1:
                // histc with auto min/max (both 0)
                result = torch::histc(input, bins, 0, 0);
                break;
            case 2:
                // histc with min > max (edge case)
                result = torch::histc(input, bins, max_val, min_val);
                break;
            case 3:
                // histc with very large bins
                result = torch::histc(input, 100000, min_val, max_val);
                break;
            case 4:
                // histc with bins = 1
                result = torch::histc(input, 1, min_val, max_val);
                break;
            case 5:
                // Test with pre-allocated output tensor
                {
                    torch::Tensor out = torch::empty({static_cast<int64_t>(bins)}, 
                                                    torch::TensorOptions().dtype(torch::kFloat32));
                    result = torch::histc_out(out, input, bins, min_val, max_val);
                }
                break;
        }
        
        // Verify result properties
        if (result.defined()) {
            // Access result to ensure computation completed
            auto sum = result.sum();
            
            // Test edge cases with the result
            if (offset < size && data[offset++] % 2) {
                // Try to reshape result
                if (result.numel() > 1) {
                    result = result.view({-1});
                }
            }
        }
        
        // Test with special tensors
        if (offset < size) {
            uint8_t special_case = data[offset++] % 5;
            torch::Tensor special_input;
            
            switch (special_case) {
                case 0:
                    // Empty tensor
                    special_input = torch::empty({0});
                    break;
                case 1:
                    // Tensor with NaN
                    special_input = torch::tensor({1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f});
                    break;
                case 2:
                    // Tensor with Inf
                    special_input = torch::tensor({1.0f, std::numeric_limits<float>::infinity(), 2.0f});
                    break;
                case 3:
                    // Very large tensor values
                    special_input = torch::tensor({1e30f, -1e30f, 0.0f});
                    break;
                case 4:
                    // Single element tensor
                    special_input = torch::tensor({42.0f});
                    break;
            }
            
            if (special_input.defined()) {
                torch::Tensor special_result = torch::histc(special_input, bins, min_val, max_val);
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 8) {
        return 0;  // Need minimum bytes for configuration
    }

    try {
        // Parse configuration from fuzzer input
        size_t offset = 0;
        
        // Extract parameters for combinations
        uint8_t r_value = data[offset++] % 10 + 1;  // r in range [1, 10]
        bool with_replacement = data[offset++] & 1;
        
        // Extract tensor configuration
        uint8_t dtype_selector = data[offset++] % 6;
        uint8_t device_selector = data[offset++] % 2;
        size_t tensor_size = (data[offset++] % 20) + 1;  // Size in range [1, 20]
        
        // Ensure we have enough data for tensor values
        if (offset + tensor_size * sizeof(float) > size) {
            tensor_size = (size - offset) / sizeof(float);
            if (tensor_size == 0) {
                return 0;
            }
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kInt16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Select device
        torch::Device device = device_selector == 0 ? torch::kCPU : torch::kCPU;
        if (device_selector == 1 && torch::cuda::is_available()) {
            device = torch::kCUDA;
        }
        
        // Create input tensor
        torch::Tensor input;
        
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            // For floating point types
            std::vector<float> values;
            for (size_t i = 0; i < tensor_size; ++i) {
                if (offset + sizeof(float) <= size) {
                    float val;
                    std::memcpy(&val, data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp to reasonable range to avoid inf/nan issues
                    if (std::isfinite(val)) {
                        values.push_back(val);
                    } else {
                        values.push_back(static_cast<float>(i));
                    }
                } else {
                    values.push_back(static_cast<float>(i));
                }
            }
            input = torch::tensor(values, torch::TensorOptions().dtype(dtype).device(device));
        } else {
            // For integer types
            std::vector<int64_t> values;
            for (size_t i = 0; i < tensor_size; ++i) {
                if (offset + sizeof(int32_t) <= size) {
                    int32_t val;
                    std::memcpy(&val, data + offset, sizeof(int32_t));
                    offset += sizeof(int32_t);
                    values.push_back(static_cast<int64_t>(val % 1000));  // Limit range
                } else {
                    values.push_back(static_cast<int64_t>(i));
                }
            }
            input = torch::tensor(values, torch::TensorOptions().dtype(dtype).device(device));
        }
        
        // Ensure input is 1D
        if (input.dim() != 1) {
            input = input.flatten();
        }
        
        // Adjust r to be valid for the tensor size
        int64_t actual_r = static_cast<int64_t>(r_value);
        if (!with_replacement && actual_r > input.size(0)) {
            actual_r = input.size(0);
        }
        
        // Test edge cases
        if (offset < size) {
            uint8_t edge_case = data[offset++] % 5;
            switch (edge_case) {
                case 0:  // Empty tensor
                    input = torch::empty({0}, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 1:  // Single element
                    input = torch::ones({1}, torch::TensorOptions().dtype(dtype).device(device));
                    break;
                case 2:  // r = 0
                    actual_r = 0;
                    break;
                case 3:  // r = tensor_size
                    actual_r = input.size(0);
                    break;
                case 4:  // r > tensor_size (for with_replacement=true)
                    if (with_replacement) {
                        actual_r = input.size(0) + 1;
                    }
                    break;
            }
        }
        
        // Call torch.combinations
        torch::Tensor result;
        
        // Test different overloads
        if (offset < size && data[offset++] % 3 == 0) {
            // Call with default parameters
            result = torch::combinations(input);
        } else if (offset < size && data[offset++] % 2 == 0) {
            // Call with r parameter only
            result = torch::combinations(input, actual_r);
        } else {
            // Call with all parameters
            result = torch::combinations(input, actual_r, with_replacement);
        }
        
        // Perform some basic operations on result to ensure it's valid
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto mean = result.mean();
            auto shape = result.sizes();
            
            // Test result tensor properties
            if (result.dim() == 2) {
                auto num_combinations = result.size(0);
                auto combination_size = result.size(1);
                
                // Verify combination size matches r
                if (combination_size != actual_r && actual_r > 0) {
                    // This might indicate an issue, but we don't crash
                }
            }
        }
        
        // Test with different memory layouts if we have more data
        if (offset + 1 < size) {
            uint8_t layout_test = data[offset++] % 3;
            switch (layout_test) {
                case 0:  // Contiguous
                    if (!input.is_contiguous()) {
                        input = input.contiguous();
                        result = torch::combinations(input, actual_r, with_replacement);
                    }
                    break;
                case 1:  // Non-contiguous via transpose (if applicable)
                    if (input.size(0) > 1) {
                        auto reshaped = input.view({-1, 1}).t().squeeze();
                        if (reshaped.size(0) == input.size(0)) {
                            result = torch::combinations(reshaped, actual_r, with_replacement);
                        }
                    }
                    break;
                case 2:  // Different stride
                    if (input.size(0) > 2) {
                        auto strided = input.slice(0, 0, -1, 2);
                        result = torch::combinations(strided, std::min(actual_r, strided.size(0)), with_replacement);
                    }
                    break;
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        return 0;  // Continue fuzzing
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected errors
    } catch (...) {
        // Unknown errors
        return -1;
    }
    
    return 0;
}
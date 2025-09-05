#include <torch/torch.h>
#include <torch/nn.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// Create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t* data, size_t& offset, size_t size) {
    // Consume shape dimensions
    uint8_t num_dims = 0;
    if (!consumeBytes(data, offset, size, num_dims)) {
        return torch::randn({1});
    }
    num_dims = num_dims % 5; // Limit to max 4 dimensions
    
    std::vector<int64_t> shape;
    for (int i = 0; i < num_dims; ++i) {
        uint8_t dim_size = 0;
        if (!consumeBytes(data, offset, size, dim_size)) break;
        shape.push_back((dim_size % 10) + 1); // Size between 1-10
    }
    
    if (shape.empty()) {
        shape.push_back(1);
    }
    
    // Consume dtype selector
    uint8_t dtype_selector = 0;
    consumeBytes(data, offset, size, dtype_selector);
    
    torch::Tensor tensor;
    switch (dtype_selector % 4) {
        case 0:
            tensor = torch::randn(shape, torch::kFloat32);
            break;
        case 1:
            tensor = torch::randn(shape, torch::kFloat64);
            break;
        case 2:
            tensor = torch::ones(shape, torch::kFloat32);
            break;
        case 3:
            tensor = torch::zeros(shape, torch::kFloat32);
            break;
    }
    
    // Optionally make it require gradient
    uint8_t requires_grad = 0;
    if (consumeBytes(data, offset, size, requires_grad)) {
        if (requires_grad % 2 == 0) {
            tensor.requires_grad_(true);
        }
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        if (size < 1) return 0;
        
        size_t offset = 0;
        
        // Create ParameterList
        torch::nn::ParameterList param_list;
        
        // Determine number of parameters to add
        uint8_t num_params = 0;
        if (!consumeBytes(data, offset, size, num_params)) {
            return 0;
        }
        num_params = (num_params % 10) + 1; // 1-10 parameters
        
        // Add parameters to the list
        std::vector<torch::Tensor> params_vec;
        for (int i = 0; i < num_params; ++i) {
            torch::Tensor tensor = createTensorFromBytes(data, offset, size);
            params_vec.push_back(tensor);
            param_list->append(tensor);
        }
        
        // Test various operations
        uint8_t op_selector = 0;
        while (consumeBytes(data, offset, size, op_selector)) {
            switch (op_selector % 10) {
                case 0: {
                    // Access by index
                    uint8_t idx = 0;
                    if (consumeBytes(data, offset, size, idx) && param_list->size() > 0) {
                        idx = idx % param_list->size();
                        auto param = param_list[idx];
                        param.sum().backward();
                    }
                    break;
                }
                case 1: {
                    // Iterate through parameters
                    for (const auto& param : *param_list) {
                        auto sum = param.sum();
                        if (param.requires_grad()) {
                            sum.backward();
                        }
                    }
                    break;
                }
                case 2: {
                    // Append new parameter
                    torch::Tensor new_tensor = createTensorFromBytes(data, offset, size);
                    param_list->append(new_tensor);
                    break;
                }
                case 3: {
                    // Extend with multiple parameters
                    std::vector<torch::Tensor> extend_vec;
                    uint8_t extend_count = 0;
                    if (consumeBytes(data, offset, size, extend_count)) {
                        extend_count = (extend_count % 3) + 1;
                        for (int i = 0; i < extend_count; ++i) {
                            extend_vec.push_back(createTensorFromBytes(data, offset, size));
                        }
                        param_list->extend(extend_vec);
                    }
                    break;
                }
                case 4: {
                    // Replace parameter at index
                    if (param_list->size() > 0) {
                        uint8_t idx = 0;
                        if (consumeBytes(data, offset, size, idx)) {
                            idx = idx % param_list->size();
                            torch::Tensor new_tensor = createTensorFromBytes(data, offset, size);
                            (*param_list)[idx] = new_tensor;
                        }
                    }
                    break;
                }
                case 5: {
                    // Get size
                    size_t list_size = param_list->size();
                    (void)list_size;
                    break;
                }
                case 6: {
                    // Clear and repopulate
                    param_list = torch::nn::ParameterList();
                    uint8_t new_count = 0;
                    if (consumeBytes(data, offset, size, new_count)) {
                        new_count = (new_count % 5) + 1;
                        for (int i = 0; i < new_count; ++i) {
                            param_list->append(createTensorFromBytes(data, offset, size));
                        }
                    }
                    break;
                }
                case 7: {
                    // Test named_parameters
                    auto named_params = param_list->named_parameters();
                    for (const auto& pair : named_params) {
                        auto name = pair.key();
                        auto param = pair.value();
                        if (param.requires_grad() && param.numel() > 0) {
                            param.sum().backward();
                        }
                    }
                    break;
                }
                case 8: {
                    // Test parameters() method
                    auto params = param_list->parameters();
                    for (auto& p : params) {
                        if (p.requires_grad() && p.numel() > 0) {
                            p.mul_(2.0);
                        }
                    }
                    break;
                }
                case 9: {
                    // Test with matrix multiplication if shapes allow
                    if (param_list->size() >= 2) {
                        try {
                            auto p1 = (*param_list)[0];
                            auto p2 = (*param_list)[1];
                            if (p1.dim() == 2 && p2.dim() == 2 && 
                                p1.size(1) == p2.size(0)) {
                                auto result = torch::mm(p1, p2);
                                if (result.requires_grad()) {
                                    result.sum().backward();
                                }
                            }
                        } catch (...) {
                            // Ignore shape mismatch errors
                        }
                    }
                    break;
                }
            }
        }
        
        // Final operations to ensure coverage
        if (param_list->size() > 0) {
            // Clone the module
            auto cloned = param_list->clone();
            
            // Test zero_grad
            param_list->zero_grad();
            
            // Test to() method for device/dtype conversion
            param_list->to(torch::kFloat64);
            param_list->to(torch::kCPU);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Unknown exceptions
        return 0;
    }
    
    return 0;
}
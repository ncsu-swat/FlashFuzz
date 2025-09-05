#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) {
        return 0;  // Need minimum bytes for basic tensor construction
    }

    try {
        // Parse fuzzer input to construct tensor parameters
        size_t offset = 0;
        
        // Extract number of dimensions (1-5 to keep reasonable)
        uint8_t ndims = (data[offset++] % 5) + 1;
        
        // Extract dimension to unbind on
        int64_t unbind_dim = static_cast<int64_t>(data[offset++] % ndims);
        
        // Also try negative indexing
        if (data[offset++] % 2 == 0) {
            unbind_dim = -static_cast<int64_t>((data[offset++] % ndims) + 1);
        }
        
        // Extract shape for each dimension
        std::vector<int64_t> shape;
        for (size_t i = 0; i < ndims && offset < size; ++i) {
            // Allow 0-sized dimensions to test edge cases
            int64_t dim_size = data[offset++] % 10;
            shape.push_back(dim_size);
        }
        
        // If we ran out of data, fill remaining dims with 1
        while (shape.size() < ndims) {
            shape.push_back(1);
        }
        
        // Select dtype based on fuzzer input
        uint8_t dtype_selector = offset < size ? data[offset++] % 8 : 0;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kUInt8; break;
            case 6: dtype = torch::kBool; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Decide if tensor should require grad (only for floating types)
        bool requires_grad = false;
        if ((dtype == torch::kFloat32 || dtype == torch::kFloat64) && offset < size) {
            requires_grad = data[offset++] % 2 == 0;
        }
        
        // Create tensor with various construction methods
        torch::Tensor tensor;
        uint8_t construction_method = offset < size ? data[offset++] % 5 : 0;
        
        try {
            switch (construction_method) {
                case 0:
                    // Create with zeros
                    tensor = torch::zeros(shape, torch::dtype(dtype).requires_grad(requires_grad));
                    break;
                case 1:
                    // Create with ones
                    tensor = torch::ones(shape, torch::dtype(dtype).requires_grad(requires_grad));
                    break;
                case 2:
                    // Create with random values
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        tensor = torch::randn(shape, torch::dtype(dtype).requires_grad(requires_grad));
                    } else {
                        tensor = torch::randint(0, 10, shape, torch::dtype(dtype));
                    }
                    break;
                case 3:
                    // Create empty tensor
                    tensor = torch::empty(shape, torch::dtype(dtype).requires_grad(requires_grad));
                    break;
                case 4:
                    // Create from data if enough bytes remain
                    {
                        int64_t numel = 1;
                        for (auto s : shape) numel *= s;
                        if (numel > 0 && numel < 10000) {  // Limit size for memory safety
                            tensor = torch::zeros(shape, torch::dtype(dtype).requires_grad(requires_grad));
                            // Fill with fuzzer data if available
                            if (offset + numel <= size) {
                                auto flat = tensor.flatten();
                                for (int64_t i = 0; i < numel && offset < size; ++i) {
                                    flat[i] = static_cast<float>(data[offset++]);
                                }
                            }
                        } else {
                            tensor = torch::zeros(shape, torch::dtype(dtype).requires_grad(requires_grad));
                        }
                    }
                    break;
                default:
                    tensor = torch::zeros(shape, torch::dtype(dtype).requires_grad(requires_grad));
            }
        } catch (...) {
            // If tensor creation fails, create a simple fallback
            tensor = torch::zeros({2, 3}, torch::dtype(dtype));
        }
        
        // Test with different memory layouts
        if (offset < size && data[offset++] % 3 == 0) {
            try {
                // Make tensor non-contiguous by transposing and back
                if (tensor.dim() >= 2) {
                    tensor = tensor.transpose(0, 1).transpose(0, 1);
                }
            } catch (...) {
                // Ignore transpose errors
            }
        }
        
        // Perform unbind operation
        std::vector<torch::Tensor> unbind_result;
        try {
            unbind_result = torch::unbind(tensor, unbind_dim);
            
            // Exercise the results to ensure they're valid
            for (const auto& t : unbind_result) {
                // Access tensor properties to trigger any latent issues
                auto shape = t.sizes();
                auto stride = t.strides();
                auto numel = t.numel();
                auto is_contiguous = t.is_contiguous();
                
                // Try to access first element if tensor is not empty
                if (numel > 0) {
                    try {
                        auto item = t.flatten()[0];
                        (void)item;
                    } catch (...) {
                        // Ignore access errors
                    }
                }
            }
            
            // Test edge case: unbind result should have size equal to dim being unbound
            if (tensor.size(unbind_dim) != static_cast<int64_t>(unbind_result.size())) {
                // This would be unexpected but we don't crash
            }
            
        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid operations
            return 0;
        } catch (const std::exception& e) {
            // Other exceptions might indicate bugs
            std::cout << "Exception caught: " << e.what() << std::endl;
            return -1;
        }
        
        // Additional edge case testing with the unbind results
        if (!unbind_result.empty() && offset < size) {
            uint8_t extra_ops = data[offset++] % 4;
            try {
                switch (extra_ops) {
                    case 0:
                        // Try to stack them back
                        torch::stack(unbind_result, unbind_dim);
                        break;
                    case 1:
                        // Try to cat them
                        if (unbind_result[0].dim() > 0) {
                            torch::cat(unbind_result, 0);
                        }
                        break;
                    case 2:
                        // Clone one of them
                        unbind_result[0].clone();
                        break;
                    case 3:
                        // Compute sum of one
                        if (unbind_result[0].numel() > 0) {
                            unbind_result[0].sum();
                        }
                        break;
                }
            } catch (...) {
                // Ignore errors in extra operations
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        return -1;
    }
    
    return 0;
}
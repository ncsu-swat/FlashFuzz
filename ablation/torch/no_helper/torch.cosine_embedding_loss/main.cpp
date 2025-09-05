#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& value) {
    if (size < sizeof(T)) return false;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

// Helper to create tensor from fuzzer input
torch::Tensor createTensorFromBytes(const uint8_t*& data, size_t& size) {
    // Consume parameters for tensor creation
    uint8_t rank;
    if (!consumeBytes(data, size, rank)) {
        return torch::randn({1});
    }
    rank = (rank % 4) + 1; // Limit rank to 1-4 for practical fuzzing
    
    std::vector<int64_t> shape;
    for (int i = 0; i < rank; i++) {
        uint8_t dim;
        if (!consumeBytes(data, size, dim)) {
            shape.push_back(1);
        } else {
            // Allow 0-dim tensors for edge cases, but cap at reasonable size
            shape.push_back(dim % 16);
        }
    }
    
    // Consume dtype choice
    uint8_t dtype_choice;
    if (!consumeBytes(data, size, dtype_choice)) {
        dtype_choice = 0;
    }
    
    torch::ScalarType dtype;
    switch (dtype_choice % 4) {
        case 0: dtype = torch::kFloat32; break;
        case 1: dtype = torch::kFloat64; break;
        case 2: dtype = torch::kFloat16; break;
        case 3: dtype = torch::kBFloat16; break;
        default: dtype = torch::kFloat32;
    }
    
    // Consume requires_grad flag
    uint8_t requires_grad;
    if (!consumeBytes(data, size, requires_grad)) {
        requires_grad = 0;
    }
    
    // Create tensor with consumed data
    torch::TensorOptions options = torch::TensorOptions().dtype(dtype).requires_grad(requires_grad % 2);
    
    // Consume initialization method
    uint8_t init_method;
    if (!consumeBytes(data, size, init_method)) {
        init_method = 0;
    }
    
    torch::Tensor tensor;
    try {
        switch (init_method % 6) {
            case 0:
                tensor = torch::randn(shape, options);
                break;
            case 1:
                tensor = torch::zeros(shape, options);
                break;
            case 2:
                tensor = torch::ones(shape, options);
                break;
            case 3:
                tensor = torch::empty(shape, options);
                break;
            case 4: {
                float value;
                consumeBytes(data, size, value);
                tensor = torch::full(shape, value, options);
                break;
            }
            case 5:
                tensor = torch::rand(shape, options);
                break;
        }
    } catch (...) {
        // Fallback to simple tensor if creation fails
        tensor = torch::randn({1}, options);
    }
    
    return tensor;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    
    try {
        const uint8_t* current_data = data;
        size_t remaining_size = size;
        
        // Create input tensors
        torch::Tensor input1 = createTensorFromBytes(current_data, remaining_size);
        torch::Tensor input2 = createTensorFromBytes(current_data, remaining_size);
        torch::Tensor target = createTensorFromBytes(current_data, remaining_size);
        
        // Consume margin parameter
        float margin = 0.0;
        if (consumeBytes(current_data, remaining_size, margin)) {
            // Allow various margin values including negative
            margin = std::fmod(margin, 10.0f);
        }
        
        // Consume reduction type
        uint8_t reduction_type;
        int64_t reduction = torch::Reduction::Mean;
        if (consumeBytes(current_data, remaining_size, reduction_type)) {
            switch (reduction_type % 3) {
                case 0: reduction = torch::Reduction::None; break;
                case 1: reduction = torch::Reduction::Mean; break;
                case 2: reduction = torch::Reduction::Sum; break;
            }
        }
        
        // Try to invoke cosine_embedding_loss with various configurations
        try {
            // Standard call
            auto result = torch::cosine_embedding_loss(
                input1, input2, target, margin, reduction
            );
            
            // Try with broadcasting scenarios
            if (remaining_size > 0) {
                uint8_t broadcast_scenario;
                consumeBytes(current_data, remaining_size, broadcast_scenario);
                
                switch (broadcast_scenario % 4) {
                    case 0:
                        // Try with mismatched dimensions
                        result = torch::cosine_embedding_loss(
                            input1.reshape({-1}), 
                            input2.reshape({-1}), 
                            target.reshape({-1}),
                            margin, reduction
                        );
                        break;
                    case 1:
                        // Try with expanded tensors
                        if (input1.numel() > 0 && input2.numel() > 0) {
                            result = torch::cosine_embedding_loss(
                                input1.expand_as(input2), 
                                input2, 
                                target,
                                margin, reduction
                            );
                        }
                        break;
                    case 2:
                        // Try with transposed tensors
                        if (input1.dim() >= 2) {
                            result = torch::cosine_embedding_loss(
                                input1.transpose(0, -1), 
                                input2.transpose(0, -1) if input2.dim() >= 2 else input2, 
                                target,
                                margin, reduction
                            );
                        }
                        break;
                    case 3:
                        // Try with contiguous vs non-contiguous
                        result = torch::cosine_embedding_loss(
                            input1.contiguous(), 
                            input2.contiguous(), 
                            target.contiguous(),
                            margin, reduction
                        );
                        break;
                }
            }
            
            // Exercise the result tensor
            if (result.defined()) {
                result.sum();
                if (result.requires_grad()) {
                    result.backward();
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
        } catch (const std::runtime_error& e) {
            // Runtime errors from shape mismatches etc.
        }
        
        // Try edge cases with special values
        if (remaining_size > 0) {
            uint8_t edge_case;
            consumeBytes(current_data, remaining_size, edge_case);
            
            switch (edge_case % 5) {
                case 0: {
                    // Empty tensors
                    auto empty1 = torch::empty({0});
                    auto empty2 = torch::empty({0});
                    auto empty_target = torch::empty({0});
                    try {
                        torch::cosine_embedding_loss(empty1, empty2, empty_target, margin, reduction);
                    } catch (...) {}
                    break;
                }
                case 1: {
                    // Inf/NaN values
                    auto inf_tensor = torch::full({2, 3}, std::numeric_limits<float>::infinity());
                    auto nan_tensor = torch::full({2, 3}, std::numeric_limits<float>::quiet_NaN());
                    auto normal_target = torch::ones({2, 3});
                    try {
                        torch::cosine_embedding_loss(inf_tensor, nan_tensor, normal_target, margin, reduction);
                    } catch (...) {}
                    break;
                }
                case 2: {
                    // Very large/small values
                    auto large = torch::full({3}, 1e30f);
                    auto small = torch::full({3}, 1e-30f);
                    auto target_val = torch::ones({3});
                    try {
                        torch::cosine_embedding_loss(large, small, target_val, margin, reduction);
                    } catch (...) {}
                    break;
                }
                case 3: {
                    // Mixed dtypes (will likely fail but tests error handling)
                    auto int_tensor = torch::randint(0, 10, {4}, torch::kInt32);
                    auto float_tensor = torch::randn({4});
                    auto target_mixed = torch::ones({4});
                    try {
                        torch::cosine_embedding_loss(int_tensor.to(torch::kFloat), float_tensor, target_mixed, margin, reduction);
                    } catch (...) {}
                    break;
                }
                case 4: {
                    // Scalar tensors
                    auto scalar1 = torch::tensor(1.5f);
                    auto scalar2 = torch::tensor(2.5f);
                    auto scalar_target = torch::tensor(1.0f);
                    try {
                        torch::cosine_embedding_loss(scalar1, scalar2, scalar_target, margin, reduction);
                    } catch (...) {}
                    break;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions to keep fuzzer running
        return -1;
    }
    
    return 0;
}
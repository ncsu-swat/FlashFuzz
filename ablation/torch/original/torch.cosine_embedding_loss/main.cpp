#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consume_value(const uint8_t* data, size_t& offset, size_t size, T default_val) {
    if (offset + sizeof(T) > size) {
        return default_val;
    }
    T val;
    std::memcpy(&val, data + offset, sizeof(T));
    offset += sizeof(T);
    return val;
}

// Helper to parse reduction mode
torch::Reduction::Reduction parse_reduction(uint8_t byte) {
    switch (byte % 3) {
        case 0: return torch::Reduction::None;
        case 1: return torch::Reduction::Mean;
        case 2: return torch::Reduction::Sum;
        default: return torch::Reduction::Mean;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {  // Need minimum bytes for basic parsing
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(data, size, offset);
        
        // Create second input tensor - try to match shape of input1 or create independently
        torch::Tensor input2;
        if (offset < size && data[offset] % 2 == 0) {
            // 50% chance: create tensor with same shape as input1
            offset++;
            if (input1.dim() > 0) {
                // Parse dtype for input2
                if (offset < size) {
                    auto dtype = fuzzer_utils::parseDataType(data[offset++]);
                    // Create tensor with same shape but potentially different dtype
                    auto options = torch::TensorOptions().dtype(dtype);
                    input2 = torch::randn(input1.sizes(), options);
                } else {
                    input2 = torch::randn_like(input1);
                }
            } else {
                // Scalar case
                input2 = fuzzer_utils::createTensor(data, size, offset);
            }
        } else {
            // 50% chance: create independent tensor
            if (offset < size) offset++;
            input2 = fuzzer_utils::createTensor(data, size, offset);
        }
        
        // Create target tensor
        torch::Tensor target;
        if (offset < size && data[offset] % 3 == 0) {
            // 33% chance: create custom target tensor
            offset++;
            target = fuzzer_utils::createTensor(data, size, offset);
            // Clamp values to -1 or 1
            target = torch::sign(target);
            target = torch::where(target.eq(0), torch::ones_like(target), target);
        } else if (offset < size && data[offset] % 3 == 1) {
            // 33% chance: all +1
            offset++;
            if (input1.dim() > 0 && input1.size(0) > 0) {
                target = torch::ones({input1.size(0)}, torch::kFloat32);
            } else {
                target = torch::ones({1}, torch::kFloat32);
            }
        } else {
            // 33% chance: all -1
            if (offset < size) offset++;
            if (input1.dim() > 0 && input1.size(0) > 0) {
                target = -torch::ones({input1.size(0)}, torch::kFloat32);
            } else {
                target = -torch::ones({1}, torch::kFloat32);
            }
        }
        
        // Parse margin value
        float margin = 0.0f;
        if (offset < size) {
            uint8_t margin_byte = data[offset++];
            // Map to range [-2.0, 2.0]
            margin = (margin_byte / 127.5f) - 1.0f;
            margin *= 2.0f;
        }
        
        // Parse reduction mode
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < size) {
            reduction = parse_reduction(data[offset++]);
        }
        
        // Try different tensor configurations
        if (offset < size && data[offset] % 4 == 0) {
            // Test with broadcasting scenarios
            offset++;
            if (input1.dim() > 1) {
                input1 = input1.squeeze();
            }
            if (input2.dim() > 1) {
                input2 = input2.squeeze();
            }
        }
        
        // Ensure tensors are floating point for cosine_embedding_loss
        if (!input1.is_floating_point()) {
            input1 = input1.to(torch::kFloat32);
        }
        if (!input2.is_floating_point()) {
            input2 = input2.to(torch::kFloat32);
        }
        if (!target.is_floating_point()) {
            target = target.to(torch::kFloat32);
        }
        
        // Test various edge cases
        if (offset < size) {
            uint8_t edge_case = data[offset++];
            
            if (edge_case % 10 == 0 && input1.numel() > 0) {
                // Add NaN values
                input1.view(-1)[0] = std::numeric_limits<float>::quiet_NaN();
            } else if (edge_case % 10 == 1 && input2.numel() > 0) {
                // Add Inf values
                input2.view(-1)[0] = std::numeric_limits<float>::infinity();
            } else if (edge_case % 10 == 2) {
                // Zero tensors
                input1.zero_();
            } else if (edge_case % 10 == 3) {
                // Very large values
                input1.mul_(1e10);
                input2.mul_(1e10);
            } else if (edge_case % 10 == 4) {
                // Very small values
                input1.mul_(1e-10);
                input2.mul_(1e-10);
            }
        }
        
        // Call cosine_embedding_loss with different configurations
        try {
            // Standard call
            auto result = torch::cosine_embedding_loss(
                input1, input2, target, 
                torch::nn::CosineEmbeddingLossOptions()
                    .margin(margin)
                    .reduction(reduction)
            );
            
            // Try with explicit options
            if (offset < size && data[offset] % 2 == 0) {
                auto options = torch::nn::CosineEmbeddingLossOptions();
                options.margin(margin);
                options.reduction(reduction);
                
                auto result2 = torch::cosine_embedding_loss(input1, input2, target, options);
            }
            
            // Test with different tensor memory layouts if possible
            if (input1.dim() > 1 && offset < size && data[offset] % 2 == 0) {
                auto input1_transposed = input1.transpose(0, 1);
                auto input2_transposed = input2.dim() > 1 ? input2.transpose(0, 1) : input2;
                
                // This might fail due to shape mismatch, which is fine for fuzzing
                auto result3 = torch::cosine_embedding_loss(
                    input1_transposed, input2_transposed, target,
                    torch::nn::CosineEmbeddingLossOptions()
                        .margin(margin)
                        .reduction(reduction)
                );
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::exception& e) {
            // Log but continue for other exceptions
        }
        
        // Additional edge case: empty tensors
        if (offset < size && data[offset] % 5 == 0) {
            torch::Tensor empty1 = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty2 = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty_target = torch::empty({0}, torch::kFloat32);
            
            try {
                auto result = torch::cosine_embedding_loss(
                    empty1, empty2, empty_target,
                    torch::nn::CosineEmbeddingLossOptions()
                        .margin(margin)
                        .reduction(reduction)
                );
            } catch (...) {
                // Expected to fail, continue
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get min and max values for clamp if we have enough data
        double min_val = -10.0;
        double max_val = 10.0;
        
        if (offset + sizeof(float) <= Size) {
            float temp;
            std::memcpy(&temp, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Avoid extreme values that might cause issues
            if (std::isfinite(temp)) {
                min_val = static_cast<double>(temp);
            }
        }
        
        if (offset + sizeof(float) <= Size) {
            float temp;
            std::memcpy(&temp, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(temp)) {
                max_val = static_cast<double>(temp);
            }
        }
        
        // Ensure min <= max
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Test different variants of clamp_
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0:
                    // clamp_ with both min and max as scalars
                    tensor.clamp_(min_val, max_val);
                    break;
                    
                case 1:
                    // clamp_min_ - clamp with only min
                    tensor.clamp_min_(min_val);
                    break;
                    
                case 2:
                    // clamp_max_ - clamp with only max
                    tensor.clamp_max_(max_val);
                    break;
                    
                case 3:
                    // clamp_ using Scalar type explicitly
                    tensor.clamp_(at::Scalar(min_val), at::Scalar(max_val));
                    break;
            }
        } else {
            // Default case: clamp_ with both min and max
            tensor.clamp_(min_val, max_val);
        }
        
        // Test with tensor arguments if we have enough data
        if (offset + 4 < Size) {
            // Create a fresh tensor for tensor-based clamp tests
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create min tensor as scalar tensor (always broadcastable)
            torch::Tensor min_tensor = torch::tensor(min_val, tensor2.options());
            
            // Create max tensor as scalar tensor (always broadcastable)
            torch::Tensor max_tensor = torch::tensor(max_val, tensor2.options());
            
            // Test clamp_ with tensor arguments
            if (offset < Size) {
                uint8_t tensor_variant = Data[offset++] % 3;
                
                try {
                    switch (tensor_variant) {
                        case 0:
                            // clamp_ with both min and max tensors
                            tensor2.clamp_(min_tensor, max_tensor);
                            break;
                            
                        case 1:
                            // clamp_min_ with tensor
                            tensor2.clamp_min_(min_tensor);
                            break;
                            
                        case 2:
                            // clamp_max_ with tensor
                            tensor2.clamp_max_(max_tensor);
                            break;
                    }
                } catch (...) {
                    // Silently handle shape/dtype mismatches
                }
            }
        }
        
        // Test with different tensor shapes (broadcastable)
        if (offset + 2 < Size && tensor.numel() > 0) {
            try {
                // Create tensors that should broadcast
                auto shape = tensor.sizes();
                torch::Tensor min_broad = torch::full({1}, min_val, tensor.options());
                torch::Tensor max_broad = torch::full({1}, max_val, tensor.options());
                
                torch::Tensor tensor_copy = tensor.clone();
                tensor_copy.clamp_(min_broad, max_broad);
            } catch (...) {
                // Silently handle any broadcast issues
            }
        }
        
        // Test edge cases with special values on floating point tensors
        if (tensor.is_floating_point() && offset < Size) {
            uint8_t special_case = Data[offset++] % 4;
            
            torch::Tensor tensor_copy = tensor.clone();
            
            try {
                switch (special_case) {
                    case 0:
                        // Very small range
                        tensor_copy.clamp_(-0.001, 0.001);
                        break;
                        
                    case 1:
                        // Zero range (all values become the same)
                        tensor_copy.clamp_(0.0, 0.0);
                        break;
                        
                    case 2:
                        // Large range
                        tensor_copy.clamp_(-1e10, 1e10);
                        break;
                        
                    case 3:
                        // Negative range
                        tensor_copy.clamp_(-100.0, -1.0);
                        break;
                }
            } catch (...) {
                // Silently handle any issues
            }
        }
        
        // Test clamp_ on integer tensors
        if (offset + 4 < Size) {
            try {
                torch::Tensor int_tensor = torch::randint(-100, 100, {4, 4}, torch::kInt32);
                int_tensor.clamp_(-50, 50);
                
                // Also test clamp_min_ and clamp_max_ on integers
                torch::Tensor int_tensor2 = torch::randint(-100, 100, {4, 4}, torch::kInt64);
                int_tensor2.clamp_min_(-25);
                int_tensor2.clamp_max_(25);
            } catch (...) {
                // Silently handle any issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
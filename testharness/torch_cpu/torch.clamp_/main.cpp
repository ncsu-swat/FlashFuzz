#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        double min_val = std::numeric_limits<double>::lowest();
        double max_val = std::numeric_limits<double>::max();
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Ensure min <= max
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Test different variants of clamp_
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // clamp_ with both min and max
                    tensor.clamp_(min_val, max_val);
                    break;
                    
                case 1:
                    // clamp_ with only min (max=None)
                    tensor.clamp_(min_val, std::nullopt);
                    break;
                    
                case 2:
                    // clamp_ with only max (min=None)
                    tensor.clamp_(std::nullopt, max_val);
                    break;
            }
        } else {
            // Default case: clamp_ with both min and max
            tensor.clamp_(min_val, max_val);
        }
        
        // Test with tensor arguments if we have enough data
        if (offset + 4 < Size) {
            // Create min tensor
            torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to match shapes if needed
            if (min_tensor.sizes() != tensor.sizes()) {
                try {
                    // Attempt to broadcast or expand
                    if (min_tensor.dim() < tensor.dim()) {
                        min_tensor = min_tensor.expand_as(tensor);
                    }
                } catch (...) {
                    // If expansion fails, create a scalar tensor
                    min_tensor = torch::tensor(min_val, tensor.options());
                }
            }
            
            // Create max tensor
            torch::Tensor max_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to match shapes if needed
            if (max_tensor.sizes() != tensor.sizes()) {
                try {
                    // Attempt to broadcast or expand
                    if (max_tensor.dim() < tensor.dim()) {
                        max_tensor = max_tensor.expand_as(tensor);
                    }
                } catch (...) {
                    // If expansion fails, create a scalar tensor
                    max_tensor = torch::tensor(max_val, tensor.options());
                }
            }
            
            // Test clamp_ with tensor arguments
            if (offset < Size) {
                uint8_t tensor_variant = Data[offset++] % 3;
                
                switch (tensor_variant) {
                    case 0:
                        // clamp_ with both min and max tensors
                        tensor.clamp_(min_tensor, max_tensor);
                        break;
                        
                    case 1:
                        // clamp_ with only min tensor
                        tensor.clamp_(min_tensor);
                        break;
                        
                    case 2:
                        // clamp_ with min tensor and max tensor (not mixed scalar/tensor)
                        tensor.clamp_(min_tensor, max_tensor);
                        break;
                }
            }
        }
        
        // Test edge cases with NaN and Inf values if tensor is floating point
        if (tensor.is_floating_point() && offset < Size) {
            uint8_t special_case = Data[offset++] % 4;
            
            torch::Tensor tensor_copy = tensor.clone();
            
            switch (special_case) {
                case 0:
                    // NaN min, regular max
                    tensor_copy.clamp_(std::numeric_limits<double>::quiet_NaN(), max_val);
                    break;
                    
                case 1:
                    // Regular min, NaN max
                    tensor_copy.clamp_(min_val, std::numeric_limits<double>::quiet_NaN());
                    break;
                    
                case 2:
                    // -Inf min, +Inf max
                    tensor_copy.clamp_(-std::numeric_limits<double>::infinity(), 
                                      std::numeric_limits<double>::infinity());
                    break;
                    
                case 3:
                    // +Inf min, -Inf max (invalid range)
                    tensor_copy.clamp_(std::numeric_limits<double>::infinity(), 
                                      -std::numeric_limits<double>::infinity());
                    break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

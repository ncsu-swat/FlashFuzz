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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract min and max values for clipping if we have enough data
        float min_val = -std::numeric_limits<float>::infinity();
        float max_val = std::numeric_limits<float>::infinity();
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Swap if min > max to ensure valid range
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Test different variants of clip_
        
        // 1. Test clip_ with both min and max
        auto tensor_copy1 = tensor.clone();
        tensor_copy1.clip_(min_val, max_val);
        
        // 2. Test clip_ with only min (max = infinity)
        auto tensor_copy2 = tensor.clone();
        tensor_copy2.clip_(min_val);
        
        // 3. Test clip_ with only max (min = -infinity)
        auto tensor_copy3 = tensor.clone();
        if (offset + 1 <= Size) {
            bool use_none = Data[offset++] & 0x1;
            if (use_none) {
                tensor_copy3.clip_(c10::nullopt, max_val);
            }
        }
        
        // 4. Test clip_ with Tensor min/max if we have enough data
        if (offset + 4 <= Size) {
            auto tensor_copy4 = tensor.clone();
            
            // Create min/max tensors with same shape as input tensor
            torch::Tensor min_tensor, max_tensor;
            
            // Use scalar tensors or tensors with same shape based on input data
            bool use_scalar_tensors = Data[offset++] & 0x1;
            
            if (use_scalar_tensors) {
                min_tensor = torch::tensor(min_val);
                max_tensor = torch::tensor(max_val);
            } else {
                min_tensor = torch::full_like(tensor, min_val);
                max_tensor = torch::full_like(tensor, max_val);
            }
            
            // Test with tensor min/max
            tensor_copy4.clip_(min_tensor, max_tensor);
            
            // Test with only tensor min
            auto tensor_copy5 = tensor.clone();
            tensor_copy5.clip_(min_tensor);
            
            // Test with only tensor max
            auto tensor_copy6 = tensor.clone();
            bool use_none_tensor = Data[offset++] & 0x1;
            if (use_none_tensor) {
                tensor_copy6.clip_(c10::nullopt, max_tensor);
            }
        }
        
        // 5. Edge case: test with min == max
        if (offset + 1 <= Size) {
            auto tensor_copy7 = tensor.clone();
            float same_val = 0.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&same_val, Data + offset, sizeof(float));
                offset += sizeof(float);
            }
            tensor_copy7.clip_(same_val, same_val);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

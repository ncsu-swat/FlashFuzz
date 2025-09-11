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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.clone operation
        torch::Tensor cloned_tensor = input_tensor.clone();
        
        // Test memory independence by modifying the original tensor
        if (input_tensor.numel() > 0 && input_tensor.is_contiguous()) {
            // Try to modify the original tensor
            if (input_tensor.is_floating_point()) {
                input_tensor.fill_(42.0);
            } else if (input_tensor.dtype() == torch::kBool) {
                input_tensor.fill_(true);
            } else {
                input_tensor.fill_(42);
            }
            
            // Verify that cloned tensor remains unchanged
            // This should never throw, but if it does, it indicates a bug in clone
            if (input_tensor.sizes() == cloned_tensor.sizes() && 
                input_tensor.dtype() == cloned_tensor.dtype()) {
                bool tensors_equal = torch::equal(input_tensor, cloned_tensor);
                if (tensors_equal) {
                    throw std::runtime_error("Clone failed: original and cloned tensors still equal after modification");
                }
            }
        }
        
        // Test cloning with different memory formats
        if (offset + 1 < Size && input_tensor.dim() >= 2) {
            uint8_t format_selector = Data[offset++];
            
            // Select memory format based on input data
            torch::MemoryFormat memory_format;
            switch (format_selector % 3) {
                case 0:
                    memory_format = torch::MemoryFormat::Contiguous;
                    break;
                case 1:
                    memory_format = torch::MemoryFormat::ChannelsLast;
                    break;
                case 2:
                    memory_format = torch::MemoryFormat::Preserve;
                    break;
                default:
                    memory_format = torch::MemoryFormat::Contiguous;
            }
            
            // Clone with specific memory format
            torch::Tensor format_cloned = input_tensor.clone(memory_format);
        }
        
        // Test non-contiguous tensor cloning
        if (input_tensor.dim() > 0 && input_tensor.numel() > 1) {
            // Create a non-contiguous view
            std::vector<int64_t> indices;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                indices.push_back(i);
            }
            
            // Randomly permute dimensions if possible
            if (offset < Size && input_tensor.dim() > 1) {
                uint8_t swap_dim = Data[offset++] % input_tensor.dim();
                uint8_t with_dim = (swap_dim + 1) % input_tensor.dim();
                std::swap(indices[swap_dim], indices[with_dim]);
            }
            
            // Create a transposed view (non-contiguous)
            torch::Tensor transposed = input_tensor.permute(indices);
            
            // Clone the non-contiguous tensor
            torch::Tensor transposed_clone = transposed.clone();
            
            // Verify that the clone has the same shape as the transposed tensor
            if (transposed.sizes() != transposed_clone.sizes()) {
                throw std::runtime_error("Clone failed: transposed and cloned tensors have different shapes");
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

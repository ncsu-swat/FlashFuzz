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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.clone operation
        torch::Tensor cloned_tensor = input_tensor.clone();
        
        // Verify basic clone properties
        if (input_tensor.sizes() != cloned_tensor.sizes()) {
            throw std::runtime_error("Clone failed: sizes mismatch");
        }
        if (input_tensor.dtype() != cloned_tensor.dtype()) {
            throw std::runtime_error("Clone failed: dtype mismatch");
        }
        
        // Test memory independence by modifying the original tensor
        if (input_tensor.numel() > 0 && input_tensor.is_contiguous()) {
            // Store a reference to check independence
            torch::Tensor cloned_copy = cloned_tensor.clone();
            
            // Modify the original tensor
            try {
                if (input_tensor.is_floating_point()) {
                    input_tensor.fill_(42.0);
                } else if (input_tensor.dtype() == torch::kBool) {
                    input_tensor.fill_(true);
                } else {
                    input_tensor.fill_(42);
                }
            } catch (...) {
                // Some tensor types may not support fill_, ignore
            }
            
            // Verify that cloned tensor remains unchanged (memory independence)
            if (!torch::equal(cloned_tensor, cloned_copy)) {
                throw std::runtime_error("Clone failed: cloned tensor was modified when original changed");
            }
        }
        
        // Test cloning with different memory formats
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            
            try {
                torch::MemoryFormat memory_format;
                switch (format_selector % 3) {
                    case 0:
                        memory_format = torch::MemoryFormat::Contiguous;
                        break;
                    case 1:
                        // ChannelsLast requires 4D tensor
                        if (input_tensor.dim() == 4) {
                            memory_format = torch::MemoryFormat::ChannelsLast;
                        } else {
                            memory_format = torch::MemoryFormat::Contiguous;
                        }
                        break;
                    case 2:
                        memory_format = torch::MemoryFormat::Preserve;
                        break;
                    default:
                        memory_format = torch::MemoryFormat::Contiguous;
                }
                
                // Clone with specific memory format
                torch::Tensor format_cloned = input_tensor.clone(memory_format);
                
                // Verify clone succeeded
                if (format_cloned.sizes() != input_tensor.sizes()) {
                    throw std::runtime_error("Clone with memory format failed: sizes mismatch");
                }
            } catch (const c10::Error&) {
                // Expected for incompatible memory format/dimension combinations
            }
        }
        
        // Test non-contiguous tensor cloning
        if (input_tensor.dim() > 1 && input_tensor.numel() > 1) {
            try {
                // Create a transposed view (non-contiguous) by swapping first two dims
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                
                // Verify it's non-contiguous
                bool was_contiguous = transposed.is_contiguous();
                
                // Clone the non-contiguous tensor
                torch::Tensor transposed_clone = transposed.clone();
                
                // Clone should produce a contiguous tensor by default
                if (!transposed_clone.is_contiguous()) {
                    // This is actually fine, just testing
                }
                
                // Verify that the clone has the same shape as the transposed tensor
                if (transposed.sizes() != transposed_clone.sizes()) {
                    throw std::runtime_error("Clone failed: transposed and cloned tensors have different shapes");
                }
                
                // Verify values are equal
                if (!torch::equal(transposed, transposed_clone)) {
                    throw std::runtime_error("Clone failed: values differ after cloning non-contiguous tensor");
                }
            } catch (const c10::Error&) {
                // Expected for some tensor configurations
            }
        }
        
        // Test cloning sliced tensors
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            try {
                // Create a slice (may be non-contiguous)
                torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0) / 2 + 1);
                torch::Tensor sliced_clone = sliced.clone();
                
                if (sliced.sizes() != sliced_clone.sizes()) {
                    throw std::runtime_error("Clone failed: sliced tensor sizes mismatch");
                }
            } catch (const c10::Error&) {
                // Expected for some configurations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
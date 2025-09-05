#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to parse memory format from fuzzer input
torch::MemoryFormat parseMemoryFormat(uint8_t selector) {
    switch (selector % 4) {
        case 0: return torch::MemoryFormat::Contiguous;
        case 1: return torch::MemoryFormat::ChannelsLast;
        case 2: return torch::MemoryFormat::ChannelsLast3d;
        case 3: return torch::MemoryFormat::Preserve;
        default: return torch::MemoryFormat::Preserve;
    }
}

// Helper to create strided tensor from fuzzer input
torch::Tensor createStridedTensor(const uint8_t* data, size_t& offset, size_t size) {
    // First create a base tensor
    torch::Tensor base = fuzzer_utils::createTensor(data, size, offset);
    
    // If we have enough data, potentially modify strides
    if (offset + base.dim() * sizeof(int64_t) <= size && base.dim() > 0) {
        std::vector<int64_t> new_strides;
        for (int i = 0; i < base.dim(); ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t stride;
                std::memcpy(&stride, data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Ensure stride is reasonable (between 1 and 1000)
                stride = 1 + (std::abs(stride) % 1000);
                new_strides.push_back(stride);
            } else {
                new_strides.push_back(base.stride(i));
            }
        }
        
        // Try to create a view with custom strides if valid
        try {
            base = base.as_strided(base.sizes(), new_strides);
        } catch (...) {
            // If custom strides are invalid, keep original tensor
        }
    }
    
    return base;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
    if (Size < 3) {
        return 0;  // Need minimum bytes for basic operations
    }
    
    try {
        size_t offset = 0;
        
        // Parse control bytes
        uint8_t memory_format_selector = Data[offset++];
        uint8_t requires_grad = Data[offset++];
        uint8_t use_strided = Data[offset++];
        
        // Create input tensor
        torch::Tensor input;
        if (use_strided % 2 == 0) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = createStridedTensor(Data, Size, offset);
        }
        
        // Set requires_grad based on fuzzer input and dtype support
        if ((requires_grad % 2 == 0) && input.dtype().isFloatingPoint()) {
            input.requires_grad_(true);
        }
        
        // Parse memory format
        torch::MemoryFormat memory_format = parseMemoryFormat(memory_format_selector);
        
        // Test clone with different memory formats
        torch::Tensor cloned;
        
        // Main clone operation with specified memory format
        try {
            cloned = input.clone(memory_format);
            
            // Verify clone properties
            if (cloned.data_ptr() == input.data_ptr()) {
                std::cerr << "Error: Clone shares data pointer with original!" << std::endl;
                return -1;
            }
            
            // Test that modifications don't affect original
            if (cloned.numel() > 0) {
                torch::Tensor cloned_copy = cloned.clone();
                cloned.fill_(0);
                if (torch::equal(input, cloned) && input.numel() > 0) {
                    std::cerr << "Error: Modifying clone affected original!" << std::endl;
                    return -1;
                }
                cloned = cloned_copy;  // Restore for further tests
            }
            
        } catch (const c10::Error& e) {
            // Some memory format conversions may not be supported for certain tensor shapes
            // Try with preserve format as fallback
            cloned = input.clone(torch::MemoryFormat::Preserve);
        }
        
        // Test gradient flow if applicable
        if (input.requires_grad()) {
            try {
                torch::Tensor loss = cloned.sum();
                loss.backward();
                
                if (!input.grad().defined()) {
                    std::cerr << "Error: Gradient not propagated to input!" << std::endl;
                    return -1;
                }
            } catch (...) {
                // Gradient computation might fail for some edge cases
            }
        }
        
        // Test clone of zero-element tensors
        if (input.numel() == 0) {
            torch::Tensor zero_clone = input.clone();
            if (zero_clone.numel() != 0) {
                std::cerr << "Error: Zero-element tensor clone has wrong size!" << std::endl;
                return -1;
            }
        }
        
        // Test clone with different memory formats if tensor supports it
        if (input.dim() == 4 && offset < Size) {
            // Try channels last for 4D tensors
            try {
                torch::Tensor cl_clone = input.clone(torch::MemoryFormat::ChannelsLast);
                if (!cl_clone.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                    std::cerr << "Warning: ChannelsLast clone not in expected format" << std::endl;
                }
            } catch (...) {
                // Some 4D tensors might not support channels last
            }
        }
        
        // Test clone of non-contiguous tensors
        if (input.dim() >= 2 && input.size(0) > 1 && input.size(1) > 1) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor trans_clone = transposed.clone();
                
                if (!trans_clone.is_contiguous() && memory_format == torch::MemoryFormat::Preserve) {
                    std::cerr << "Warning: Clone of transposed tensor became contiguous unexpectedly" << std::endl;
                }
            } catch (...) {
                // Transpose might fail for some tensor configurations
            }
        }
        
        // Test clone with slicing
        if (input.dim() > 0 && input.size(0) > 2) {
            try {
                torch::Tensor sliced = input.narrow(0, 0, 2);
                torch::Tensor slice_clone = sliced.clone();
                
                if (slice_clone.size(0) != 2) {
                    std::cerr << "Error: Sliced clone has wrong size!" << std::endl;
                    return -1;
                }
            } catch (...) {
                // Narrow might fail for some configurations
            }
        }
        
        // Test multiple clones
        if (offset + 1 < Size) {
            uint8_t num_clones = Data[offset++] % 5 + 1;
            std::vector<torch::Tensor> clones;
            
            for (int i = 0; i < num_clones; ++i) {
                clones.push_back(input.clone());
            }
            
            // Verify all clones are independent
            for (size_t i = 0; i < clones.size(); ++i) {
                for (size_t j = i + 1; j < clones.size(); ++j) {
                    if (clones[i].data_ptr() == clones[j].data_ptr()) {
                        std::cerr << "Error: Multiple clones share data!" << std::endl;
                        return -1;
                    }
                }
            }
        }
        
        // Test clone of views
        if (input.dim() >= 1 && input.numel() > 0) {
            try {
                torch::Tensor view = input.view({-1});
                torch::Tensor view_clone = view.clone();
                
                if (view_clone.dim() != 1) {
                    std::cerr << "Error: Clone of view has wrong dimensions!" << std::endl;
                    return -1;
                }
            } catch (...) {
                // View might fail for some configurations
            }
        }
        
        // Test clone preserves dtype
        if (cloned.dtype() != input.dtype()) {
            std::cerr << "Error: Clone changed dtype!" << std::endl;
            return -1;
        }
        
        // Test clone preserves device (CPU in this case)
        if (cloned.device() != input.device()) {
            std::cerr << "Error: Clone changed device!" << std::endl;
            return -1;
        }
        
    } catch (const std::exception& e) {
        // Log specific exceptions for debugging but don't crash
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other exceptions
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}
#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (size < 4) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(data, size, offset);
        } catch (const std::exception& e) {
            // If we can't create a valid input tensor, skip this input
            return 0;
        }

        // Parse additional options if we have remaining bytes
        torch::TensorOptions options;
        bool has_dtype = false;
        bool has_layout = false;
        bool has_device = false;
        bool requires_grad = false;
        c10::optional<torch::MemoryFormat> memory_format = c10::nullopt;

        if (offset < size) {
            uint8_t config_byte = data[offset++];
            
            // Bit 0: whether to specify dtype
            if (config_byte & 0x01 && offset < size) {
                has_dtype = true;
                uint8_t dtype_selector = data[offset++];
                try {
                    auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                    options = options.dtype(dtype);
                } catch (...) {
                    // Invalid dtype, continue without it
                    has_dtype = false;
                }
            }
            
            // Bit 1: whether to specify layout
            if (config_byte & 0x02 && offset < size) {
                has_layout = true;
                uint8_t layout_selector = data[offset++];
                // PyTorch supports kStrided and kSparse
                if (layout_selector % 2 == 0) {
                    options = options.layout(torch::kStrided);
                } else {
                    // Sparse layout requires special handling, skip for rand_like
                    // as it doesn't support sparse tensors directly
                    has_layout = false;
                }
            }
            
            // Bit 2: whether to specify device
            if (config_byte & 0x04 && offset < size) {
                has_device = true;
                uint8_t device_selector = data[offset++];
                // Test CPU and CUDA (if available)
                if (torch::cuda::is_available() && device_selector % 2 == 1) {
                    options = options.device(torch::kCUDA);
                } else {
                    options = options.device(torch::kCPU);
                }
            }
            
            // Bit 3: requires_grad
            if (config_byte & 0x08) {
                requires_grad = true;
                options = options.requires_grad(true);
            }
            
            // Bit 4: whether to specify memory format
            if (config_byte & 0x10 && offset < size) {
                uint8_t format_selector = data[offset++];
                switch (format_selector % 4) {
                    case 0:
                        memory_format = torch::MemoryFormat::Preserve;
                        break;
                    case 1:
                        memory_format = torch::MemoryFormat::Contiguous;
                        break;
                    case 2:
                        memory_format = torch::MemoryFormat::ChannelsLast;
                        break;
                    case 3:
                        memory_format = torch::MemoryFormat::ChannelsLast3d;
                        break;
                }
            }
        }

        // Test various invocations of rand_like
        torch::Tensor result;
        
        // Basic invocation - just input tensor
        try {
            result = torch::rand_like(input_tensor);
            
            // Verify basic properties
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Size mismatch in basic rand_like" << std::endl;
            }
            
            // Check values are in [0, 1) for floating point types
            if (result.is_floating_point()) {
                auto min_val = torch::min(result);
                auto max_val = torch::max(result);
                if (min_val.item<double>() < 0.0 || max_val.item<double>() >= 1.0) {
                    std::cerr << "Values out of range [0, 1)" << std::endl;
                }
            }
        } catch (const c10::Error& e) {
            // Some configurations might not be supported
            // Continue testing other configurations
        }
        
        // Test with explicit dtype
        if (has_dtype) {
            try {
                result = torch::rand_like(input_tensor, options);
                
                // Verify dtype was applied
                if (has_dtype && result.dtype() != options.dtype()) {
                    std::cerr << "Dtype not correctly applied" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some dtype conversions might not be supported
            }
        }
        
        // Test with memory format if specified
        if (memory_format.has_value()) {
            try {
                // Create a new options object for this test
                torch::TensorOptions mem_options = input_tensor.options();
                if (has_dtype) {
                    mem_options = mem_options.dtype(options.dtype());
                }
                
                result = torch::rand_like(input_tensor, mem_options, memory_format.value());
                
                // For 4D tensors, check if channels_last format was applied
                if (memory_format.value() == torch::MemoryFormat::ChannelsLast && 
                    input_tensor.dim() == 4) {
                    if (!result.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                        std::cerr << "ChannelsLast format not applied" << std::endl;
                    }
                }
            } catch (const c10::Error& e) {
                // Memory format might not be applicable to all tensor shapes
            }
        }
        
        // Test edge cases with special tensor properties
        try {
            // Test with zero-size tensor dimensions
            if (input_tensor.numel() == 0) {
                result = torch::rand_like(input_tensor);
                if (result.numel() != 0) {
                    std::cerr << "Zero-size tensor not preserved" << std::endl;
                }
            }
            
            // Test with scalar tensor (0-dimensional)
            if (input_tensor.dim() == 0) {
                result = torch::rand_like(input_tensor);
                if (result.dim() != 0) {
                    std::cerr << "Scalar tensor dimension not preserved" << std::endl;
                }
            }
            
            // Test with requires_grad
            if (requires_grad) {
                torch::TensorOptions grad_options = input_tensor.options().requires_grad(true);
                result = torch::rand_like(input_tensor, grad_options);
                if (!result.requires_grad()) {
                    std::cerr << "requires_grad not set" << std::endl;
                }
            }
            
            // Test multiple calls for consistency in shape
            torch::Tensor result2 = torch::rand_like(input_tensor);
            if (result2.sizes() != input_tensor.sizes()) {
                std::cerr << "Inconsistent shape in multiple calls" << std::endl;
            }
            
            // Values should be different (probabilistically)
            if (input_tensor.numel() > 0 && result.is_floating_point() && result2.is_floating_point()) {
                if (torch::equal(result, result2)) {
                    // This is extremely unlikely for random values
                    std::cerr << "Suspicious: Two rand_like calls produced identical values" << std::endl;
                }
            }
            
        } catch (const c10::Error& e) {
            // Some edge cases might fail, continue
        }
        
        // Test with different tensor types and properties
        try {
            // Test with non-contiguous tensor
            if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
                auto transposed = input_tensor.transpose(0, 1);
                if (!transposed.is_contiguous()) {
                    result = torch::rand_like(transposed);
                    if (result.sizes() != transposed.sizes()) {
                        std::cerr << "Non-contiguous tensor shape not preserved" << std::endl;
                    }
                }
            }
            
            // Test with view
            if (input_tensor.numel() > 0) {
                auto viewed = input_tensor.view({-1});
                result = torch::rand_like(viewed);
                if (result.numel() != viewed.numel()) {
                    std::cerr << "View tensor element count not preserved" << std::endl;
                }
            }
            
        } catch (const c10::Error& e) {
            // View or transpose operations might fail for certain shapes
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const c10::Error &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}
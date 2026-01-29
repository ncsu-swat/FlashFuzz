#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply basic randn_like operation
        torch::Tensor output_tensor = torch::randn_like(input_tensor);
        
        // Verify the output tensor has the same shape as the input
        if (input_tensor.sizes() != output_tensor.sizes()) {
            throw std::runtime_error("Output tensor shape doesn't match input tensor shape");
        }
        
        // Try with different options based on fuzzer data
        if (offset + 2 < Size) {
            uint8_t option_byte = Data[offset++];
            uint8_t param_byte = Data[offset++];
            
            // Try with different dtype (must be floating point for randn)
            if (option_byte & 0x01) {
                torch::ScalarType dtype;
                switch (param_byte % 4) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kFloat16; break;
                    default: dtype = torch::kFloat32; break;
                }
                try {
                    output_tensor = torch::randn_like(input_tensor, 
                        torch::TensorOptions().dtype(dtype));
                } catch (...) {
                    // Some dtype conversions may fail, ignore silently
                }
            }
            
            // Try with device option (CPU only in this harness)
            if (option_byte & 0x02) {
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().device(torch::kCPU));
            }
            
            // Try with requires_grad (only valid for floating point output)
            if (option_byte & 0x04) {
                bool requires_grad = (param_byte % 2 == 0);
                try {
                    // randn_like produces float by default, so requires_grad should work
                    output_tensor = torch::randn_like(input_tensor, 
                        torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .requires_grad(requires_grad));
                } catch (...) {
                    // Silently ignore if requires_grad fails
                }
            }
            
            // Try with memory format - Contiguous or Preserve only (safest options)
            if (option_byte & 0x08) {
                torch::MemoryFormat memory_format = (param_byte % 2 == 0) ? 
                    torch::MemoryFormat::Contiguous : torch::MemoryFormat::Preserve;
                output_tensor = torch::randn_like(input_tensor, 
                    torch::TensorOptions().memory_format(memory_format));
            }
            
            // Try ChannelsLast only with 4D tensors
            if ((option_byte & 0x10) && input_tensor.dim() == 4) {
                try {
                    output_tensor = torch::randn_like(input_tensor, 
                        torch::TensorOptions().memory_format(torch::MemoryFormat::ChannelsLast));
                } catch (...) {
                    // May fail for certain tensor configurations
                }
            }
            
            // Try ChannelsLast3d only with 5D tensors
            if ((option_byte & 0x20) && input_tensor.dim() == 5) {
                try {
                    output_tensor = torch::randn_like(input_tensor, 
                        torch::TensorOptions().memory_format(torch::MemoryFormat::ChannelsLast3d));
                } catch (...) {
                    // May fail for certain tensor configurations
                }
            }
            
            // Try with combined options - float dtype with requires_grad
            if (option_byte & 0x40) {
                torch::ScalarType dtype;
                switch (param_byte % 3) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    default: dtype = torch::kFloat32; break;
                }
                bool requires_grad = ((param_byte >> 2) % 2 == 0);
                try {
                    output_tensor = torch::randn_like(input_tensor, 
                        torch::TensorOptions()
                            .dtype(dtype)
                            .device(torch::kCPU)
                            .requires_grad(requires_grad));
                } catch (...) {
                    // Silently ignore failures
                }
            }
            
            // Test that output is always the same shape
            if (input_tensor.sizes() != output_tensor.sizes()) {
                throw std::runtime_error("Output tensor shape mismatch after options");
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
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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors to avoid issues with min/max checks
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        torch::Tensor result;
        
        if (offset + 1 < Size) {
            uint8_t options_byte = Data[offset++];
            
            // Get a data type for the output tensor - only use floating point types
            // since rand_like requires floating point
            torch::ScalarType output_dtype;
            uint8_t dtype_selector = (offset < Size) ? (Data[offset++] % 3) : 0;
            switch (dtype_selector) {
                case 0:
                    output_dtype = torch::kFloat32;
                    break;
                case 1:
                    output_dtype = torch::kFloat64;
                    break;
                default:
                    output_dtype = torch::kFloat16;
                    break;
            }
            
            // requires_grad only valid for floating point
            bool requires_grad = (options_byte & 0x01) && 
                                 (output_dtype == torch::kFloat32 || 
                                  output_dtype == torch::kFloat64);
            
            // Create tensor options (skip pinned_memory as it requires CUDA)
            auto tensor_options = torch::TensorOptions()
                .dtype(output_dtype)
                .requires_grad(requires_grad);
            
            // Test different variants of rand_like
            if (offset < Size) {
                uint8_t variant = Data[offset++] % 3;
                
                switch (variant) {
                    case 0:
                        // Basic rand_like
                        result = torch::rand_like(input_tensor);
                        break;
                    case 1:
                        // rand_like with options
                        result = torch::rand_like(input_tensor, tensor_options);
                        break;
                    case 2:
                        // rand_like with memory format
                        {
                            uint8_t format_byte = (offset < Size) ? (Data[offset++] % 4) : 0;
                            c10::MemoryFormat memory_format;
                            
                            switch (format_byte) {
                                case 0:
                                    memory_format = c10::MemoryFormat::Contiguous;
                                    break;
                                case 1:
                                    // ChannelsLast requires 4D tensor
                                    if (input_tensor.dim() == 4) {
                                        memory_format = c10::MemoryFormat::ChannelsLast;
                                    } else {
                                        memory_format = c10::MemoryFormat::Contiguous;
                                    }
                                    break;
                                case 2:
                                    // ChannelsLast3d requires 5D tensor
                                    if (input_tensor.dim() == 5) {
                                        memory_format = c10::MemoryFormat::ChannelsLast3d;
                                    } else {
                                        memory_format = c10::MemoryFormat::Contiguous;
                                    }
                                    break;
                                default:
                                    memory_format = c10::MemoryFormat::Preserve;
                                    break;
                            }
                            
                            result = torch::rand_like(input_tensor, tensor_options.memory_format(memory_format));
                        }
                        break;
                }
            } else {
                result = torch::rand_like(input_tensor);
            }
        } else {
            // Basic case with default options
            result = torch::rand_like(input_tensor);
        }
        
        // Basic validation - check that result has same shape as input
        if (result.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("rand_like produced tensor with different shape");
        }
        
        // Check that values are in the expected range [0, 1) for non-empty float tensors
        if (result.numel() > 0 && result.is_floating_point()) {
            // Convert to float for comparison
            auto result_float = result.to(torch::kFloat32);
            auto min_val = torch::min(result_float).item<float>();
            auto max_val = torch::max(result_float).item<float>();
            
            if (min_val < 0.0f || max_val >= 1.0f) {
                throw std::runtime_error("rand_like produced values outside [0, 1) range");
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
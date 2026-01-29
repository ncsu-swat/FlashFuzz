#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply zeros_like operation
        torch::Tensor result = torch::zeros_like(input_tensor);
        
        // Try with different options - requires_grad only works with floating point
        if (offset + 1 < Size) {
            try {
                bool requires_grad = Data[offset++] & 0x01;
                // Only set requires_grad if input is floating point
                if (input_tensor.is_floating_point() && requires_grad) {
                    torch::Tensor result_with_grad = torch::zeros_like(
                        input_tensor, 
                        torch::TensorOptions().requires_grad(true)
                    );
                }
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Try with different memory format - must match tensor dimensions
        if (offset + 1 < Size) {
            try {
                uint8_t memory_format_selector = Data[offset++] % 4;
                c10::MemoryFormat memory_format;
                
                int64_t ndim = input_tensor.dim();
                
                switch (memory_format_selector) {
                    case 0:
                        memory_format = c10::MemoryFormat::Contiguous;
                        break;
                    case 1:
                        // ChannelsLast requires 4D tensor
                        if (ndim == 4) {
                            memory_format = c10::MemoryFormat::ChannelsLast;
                        } else {
                            memory_format = c10::MemoryFormat::Contiguous;
                        }
                        break;
                    case 2:
                        // ChannelsLast3d requires 5D tensor
                        if (ndim == 5) {
                            memory_format = c10::MemoryFormat::ChannelsLast3d;
                        } else {
                            memory_format = c10::MemoryFormat::Contiguous;
                        }
                        break;
                    case 3:
                        memory_format = c10::MemoryFormat::Preserve;
                        break;
                    default:
                        memory_format = c10::MemoryFormat::Contiguous;
                }
                
                torch::Tensor result_with_memory_format = torch::zeros_like(
                    input_tensor, 
                    torch::TensorOptions().memory_format(memory_format)
                );
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Try with different dtype
        if (offset + 1 < Size) {
            try {
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
                torch::Tensor result_with_dtype = torch::zeros_like(
                    input_tensor, 
                    torch::TensorOptions().dtype(dtype)
                );
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Try with different device (CPU only in this harness)
        if (offset + 1 < Size) {
            try {
                offset++; // Consume the byte but stay on CPU
                torch::Device device = torch::Device(torch::kCPU);
                    
                torch::Tensor result_with_device = torch::zeros_like(
                    input_tensor, 
                    torch::TensorOptions().device(device)
                );
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Try with multiple options combined
        if (offset + 3 < Size) {
            try {
                bool requires_grad = Data[offset++] & 0x01;
                torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
                offset++; // Consume device byte, stay on CPU
                torch::Device device = torch::Device(torch::kCPU);
                
                // Build options
                auto options = torch::TensorOptions().dtype(dtype).device(device);
                
                // Only add requires_grad for floating point types
                bool is_float_dtype = (dtype == torch::kFloat32 || dtype == torch::kFloat64 ||
                                       dtype == torch::kFloat16 || dtype == torch::kBFloat16);
                if (requires_grad && is_float_dtype) {
                    options = options.requires_grad(true);
                }
                    
                torch::Tensor result_combined = torch::zeros_like(input_tensor, options);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Verify that all elements in the result tensor are zeros
        if (result.numel() > 0) {
            try {
                bool all_zeros = torch::all(result == 0).item<bool>();
                if (!all_zeros) {
                    std::cerr << "zeros_like produced non-zero values" << std::endl;
                }
            } catch (...) {
                // Comparison might fail for certain dtypes
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
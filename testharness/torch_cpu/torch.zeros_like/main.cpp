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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply zeros_like operation
        torch::Tensor result = torch::zeros_like(input_tensor);
        
        // Try with different options
        if (offset + 1 < Size) {
            bool requires_grad = Data[offset++] & 0x01;
            torch::Tensor result_with_grad = torch::zeros_like(input_tensor, torch::TensorOptions().requires_grad(requires_grad));
        }
        
        // Try with different memory format
        if (offset + 1 < Size) {
            uint8_t memory_format_selector = Data[offset++] % 4;
            c10::MemoryFormat memory_format;
            
            switch (memory_format_selector) {
                case 0:
                    memory_format = c10::MemoryFormat::Contiguous;
                    break;
                case 1:
                    memory_format = c10::MemoryFormat::ChannelsLast;
                    break;
                case 2:
                    memory_format = c10::MemoryFormat::ChannelsLast3d;
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
        }
        
        // Try with different dtype
        if (offset + 1 < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            torch::Tensor result_with_dtype = torch::zeros_like(
                input_tensor, 
                torch::TensorOptions().dtype(dtype)
            );
        }
        
        // Try with different device
        if (offset + 1 < Size) {
            bool use_cuda = Data[offset++] & 0x01;
            torch::Device device = use_cuda && torch::cuda::is_available() ? 
                torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
                
            torch::Tensor result_with_device = torch::zeros_like(
                input_tensor, 
                torch::TensorOptions().device(device)
            );
        }
        
        // Try with multiple options combined
        if (offset + 3 < Size) {
            bool requires_grad = Data[offset++] & 0x01;
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            bool use_cuda = Data[offset++] & 0x01;
            torch::Device device = use_cuda && torch::cuda::is_available() ? 
                torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
                
            torch::Tensor result_combined = torch::zeros_like(
                input_tensor, 
                torch::TensorOptions()
                    .requires_grad(requires_grad)
                    .dtype(dtype)
                    .device(device)
            );
        }
        
        // Verify that all elements in the result tensor are zeros
        if (result.numel() > 0) {
            bool all_zeros = torch::all(result == 0).item<bool>();
            if (!all_zeros) {
                throw std::runtime_error("zeros_like produced non-zero values");
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

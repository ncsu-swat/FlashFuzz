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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get device type from the next byte if available
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            // Use only 0 (CPU) or 1 (CUDA) if available
            if (device_selector % 2 == 1 && torch::cuda::is_available()) {
                device = torch::kCUDA;
            }
        }
        
        // Get index from the next byte if available
        int64_t index = 0;
        if (offset < Size) {
            uint8_t index_byte = Data[offset++];
            index = static_cast<int64_t>(index_byte % 8); // Limit to reasonable range
        }
        
        // Test basic device operations
        
        // Get current device
        auto current_device = tensor.device();
        
        // Move tensor to different device if CUDA is available
        if (torch::cuda::is_available()) {
            auto cuda_tensor = tensor.to(torch::kCUDA);
            auto back_to_cpu = cuda_tensor.to(torch::kCPU);
        }
        
        // Test CUDA specific operations if available
        if (torch::cuda::is_available()) {
            // Get CUDA device count
            int64_t cuda_count = torch::cuda::device_count();
            
            // Get current CUDA device
            int64_t current_cuda_device = torch::cuda::current_device();
            
            // Set CUDA device (with bounds checking)
            if (cuda_count > 0) {
                torch::cuda::set_device(index % cuda_count);
                
                // Synchronize CUDA
                torch::cuda::synchronize();
                
                // Empty CUDA cache
                torch::cuda::empty_cache();
                
                // Reset to device 0
                torch::cuda::set_device(0);
            }
        }
        
        // Test tensor device placement
        auto cpu_tensor = tensor.to(torch::kCPU);
        
        // Test device properties through tensor operations
        auto tensor_device = tensor.device();
        bool is_cuda = tensor_device.is_cuda();
        bool is_cpu = tensor_device.is_cpu();
        
        // Create tensors on specific devices
        auto options_cpu = torch::TensorOptions().device(torch::kCPU);
        auto cpu_created = torch::zeros({2, 2}, options_cpu);
        
        if (torch::cuda::is_available()) {
            auto options_cuda = torch::TensorOptions().device(torch::kCUDA);
            auto cuda_created = torch::zeros({2, 2}, options_cuda);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

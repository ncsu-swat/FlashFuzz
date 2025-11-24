#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Target API: torch.backends
    (void)"torch.backends";
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with backends operations
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::cuda operations
        bool cuda_available = torch::cuda::is_available();
        if (cuda_available) {
            int device_count = torch::cuda::device_count();
            
            if (offset < Size && device_count > 0) {
                int device_idx = Data[offset++] % device_count;
                bool cudnn_available = torch::cuda::cudnn_is_available();
                if (cudnn_available) {
                    torch::cuda::manual_seed_all(static_cast<uint64_t>(Size));
                }
                
                // Exercise CUDA backend synchronization for a specific device
                torch::cuda::synchronize(device_idx);
            }
        }
        
        // Test basic tensor operations that might use different backends
        if (tensor.numel() > 0) {
            auto sum_result = tensor.sum();
            auto mean_result = tensor.mean();
            (void)sum_result;
            (void)mean_result;
            
            // Test different dtypes
            if (offset < Size) {
                auto dtype_idx = Data[offset++] % 4;
                torch::ScalarType target_dtype;
                switch (dtype_idx) {
                    case 0: target_dtype = torch::kFloat32; break;
                    case 1: target_dtype = torch::kFloat64; break;
                    case 2: target_dtype = torch::kInt32; break;
                    default: target_dtype = torch::kInt64; break;
                }
                auto converted_tensor = tensor.to(target_dtype);
                (void)converted_tensor;
            }
        }
        
        // Test device placement
        if (offset < Size) {
            bool use_cuda = Data[offset++] % 2 == 0;
            torch::Device target_device = use_cuda && torch::cuda::is_available() ? 
                torch::kCUDA : torch::kCPU;
            
            try {
                auto device_tensor = tensor.to(target_device);
                (void)device_tensor;
            } catch (const std::exception&) {
                // Ignore device placement errors
            }
        }
        
        // Test autograd functionality
        if (tensor.requires_grad()) {
            try {
                auto grad_tensor = tensor.sum();
                grad_tensor.backward();
            } catch (const std::exception&) {
                // Ignore autograd errors
            }
        }
        
        // Test memory format operations
        if (tensor.dim() == 4 && offset < Size) {
            bool use_channels_last = Data[offset++] % 2 == 0;
            try {
                auto memory_format = use_channels_last ? 
                    torch::MemoryFormat::ChannelsLast : torch::MemoryFormat::Contiguous;
                auto formatted_tensor = tensor.contiguous(memory_format);
                (void)formatted_tensor;
            } catch (const std::exception&) {
                // Ignore memory format errors
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

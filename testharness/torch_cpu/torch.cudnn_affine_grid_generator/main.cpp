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
        
        // Need at least a few bytes for the theta tensor and output size
        if (Size < 8) {
            return 0;
        }
        
        // Create the theta tensor (2x3 affine matrix)
        torch::Tensor theta = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least one more byte for batch size
        if (offset >= Size) {
            return 0;
        }
        
        // Parse batch size
        int64_t N = static_cast<int64_t>(Data[offset++]) % 16 + 1;
        
        // Parse output size dimensions
        int64_t C = 0, H = 0, W = 0;
        
        // Ensure we have enough bytes for output dimensions
        if (offset + 2 <= Size) {
            C = static_cast<int64_t>(Data[offset++]) % 8 + 1;
            H = static_cast<int64_t>(Data[offset++]) % 64 + 1;
            W = static_cast<int64_t>(Data[offset++]) % 64 + 1;
        } else {
            // Default values if not enough data
            C = 3;
            H = 32;
            W = 32;
        }
        
        // Try to reshape theta to the expected shape for the operation
        // For 2D: [N, 2, 3]
        try {
            theta = theta.reshape({N, 2, 3});
        } catch (const std::exception& e) {
            // If reshape fails, create a new tensor with the right shape
            theta = torch::rand({N, 2, 3}, theta.options());
        }
        
        // Move tensors to CUDA if available
        if (torch::cuda::is_available()) {
            theta = theta.to(torch::kCUDA);
        }
        
        // Call cudnn_affine_grid_generator
        torch::Tensor output = torch::cudnn_affine_grid_generator(theta, N, C, H, W);
        
        // Verify output shape
        // 2D case: output should be [N, H, W, 2]
        if (output.dim() != 4 || 
            output.size(0) != N || 
            output.size(1) != H || 
            output.size(2) != W || 
            output.size(3) != 2) {
            throw std::runtime_error("Unexpected output shape for 4D case");
        }
        
        // Move output back to CPU if it was on CUDA
        if (output.is_cuda()) {
            output = output.to(torch::kCPU);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

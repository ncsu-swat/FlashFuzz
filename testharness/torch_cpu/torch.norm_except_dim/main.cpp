#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Need at least a few bytes for basic tensor creation and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (require float for norm computation)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor for norm computation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure tensor has at least 1 dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Extract parameters for norm_except_dim
        // pow: the norm type (typically 1 or 2)
        int64_t pow_raw = 2;
        if (offset + sizeof(uint8_t) <= Size) {
            pow_raw = static_cast<int64_t>(Data[offset] % 3) + 1; // 1, 2, or 3
            offset += sizeof(uint8_t);
        }
        
        // dim: which dimension to exclude from norm calculation
        // Must be valid for the tensor's dimensions
        int64_t dim = 0;
        if (offset + sizeof(uint8_t) <= Size) {
            dim = static_cast<int64_t>(Data[offset]) % input.dim();
            offset += sizeof(uint8_t);
        }
        
        // Apply norm_except_dim operation
        // This function computes the norm over all dimensions except 'dim'
        torch::Tensor result = torch::norm_except_dim(input, pow_raw, dim);
        
        // Basic validation - just ensure the result tensor is valid
        volatile int64_t numel = result.numel();
        (void)numel;
        
        // Try to access sum to verify computation
        if (result.numel() > 0 && result.is_floating_point()) {
            volatile float sum_val = result.sum().item<float>();
            (void)sum_val;
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors (shape mismatches, etc.) - expected during fuzzing
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
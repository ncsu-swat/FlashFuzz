#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Need at least a few bytes for meaningful input
        if (Size < 8) {
            return 0;
        }
        
        // Create first input tensor for PairwiseDistance
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        // Create second tensor with the same shape as first to avoid shape mismatch
        torch::Tensor x2;
        try {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape x2 to match x1 if possible
            if (x1.sizes() != x2.sizes()) {
                // Try to create x2 with same shape by using remaining data
                x2 = torch::randn(x1.sizes());
                // Use fuzzer data to modify values if available
                if (offset < Size) {
                    float scale = static_cast<float>(Data[offset]) / 128.0f;
                    offset++;
                    x2 = x2 * scale;
                }
            }
        } catch (...) {
            // If tensor creation fails, create a tensor matching x1's shape
            x2 = torch::randn(x1.sizes());
        }
        
        // Get parameters for PairwiseDistance
        double p = 2.0;  // Default p-norm value
        bool keepdim = false;
        double eps = 1e-6;
        
        // Extract p value from input data if available
        if (offset + sizeof(float) <= Size) {
            float p_float;
            std::memcpy(&p_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is a valid, reasonable value
            if (!std::isnan(p_float) && !std::isinf(p_float)) {
                p = std::abs(static_cast<double>(p_float));
            }
            // Clamp p to reasonable range [0.1, 10.0]
            if (p < 0.1) p = 0.1;
            if (p > 10.0) p = 10.0;
        }
        
        // Extract keepdim from input data if available
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Extract eps value from input data if available
        if (offset + sizeof(float) <= Size) {
            float eps_float;
            std::memcpy(&eps_float, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is a valid, reasonable value
            if (!std::isnan(eps_float) && !std::isinf(eps_float)) {
                eps = std::abs(static_cast<double>(eps_float));
            }
            // Clamp eps to reasonable range
            if (eps < 1e-10) eps = 1e-10;
            if (eps > 1.0) eps = 1.0;
        }
        
        // Inner try-catch for expected operational failures
        try {
            // Create PairwiseDistance module
            torch::nn::PairwiseDistance pairwise_distance(
                torch::nn::PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim)
            );
            
            // Apply the operation
            torch::Tensor output = pairwise_distance->forward(x1, x2);
            
            // Try to access the output tensor to ensure computation is performed
            if (output.defined() && output.numel() > 0) {
                // Sum to trigger computation without issues on multi-element tensors
                auto sum_val = output.sum().item<float>();
                (void)sum_val;  // Prevent unused variable warning
            }
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (shape mismatch, etc.) - silently ignore
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least some bytes for tensor creation
        if (Size < 4)
            return 0;
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size)
            return 0;
            
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // pairwise_distance requires tensors to be broadcastable
        // and works on the last dimension (feature dimension)
        // Tensors should be at least 1D
        if (x1.dim() < 1 || x2.dim() < 1)
            return 0;
        
        // Parse p-norm value from remaining data
        double p = 2.0; // Default p-norm
        if (offset + sizeof(float) <= Size) {
            float p_value;
            std::memcpy(&p_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is positive and reasonable
            p = std::abs(static_cast<double>(p_value));
            if (p < 1e-6 || std::isnan(p) || std::isinf(p)) {
                p = 2.0; // Avoid very small/invalid values
            }
            // Clamp to reasonable range
            if (p > 100.0) p = 100.0;
        }
        
        // Parse epsilon value
        double eps = 1e-6; // Default epsilon
        if (offset + sizeof(float) <= Size) {
            float eps_value;
            std::memcpy(&eps_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is non-negative and reasonable
            eps = std::abs(static_cast<double>(eps_value));
            if (std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-6;
            }
            // Clamp epsilon to reasonable range
            if (eps > 1.0) eps = 1.0;
        }
        
        // Parse keepdim flag
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Try to make tensors broadcastable for pairwise_distance
        // The function expects inputs of shape (N, D) or broadcastable
        try {
            // Apply pairwise_distance operation
            torch::Tensor result = torch::pairwise_distance(x1, x2, p, eps, keepdim);
            
            // Ensure result is computed
            if (result.numel() > 0) {
                volatile double sum = result.sum().item<double>();
                (void)sum;
            }
        }
        catch (const c10::Error &) {
            // Shape mismatch or other expected errors - silently continue
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
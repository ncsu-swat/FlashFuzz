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
        
        // Need at least 2 tensors for pairwise_distance
        if (Size < 4) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size)
            return 0;
            
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensors have the same shape
        if (x1.sizes() != x2.sizes() || x1.dim() < 1)
            return 0;
        
        // Parse p-norm value from remaining data
        double p = 2.0; // Default p-norm
        if (offset + sizeof(float) <= Size) {
            float p_value;
            std::memcpy(&p_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is positive
            p = std::abs(p_value);
            if (p < 1e-6) p = 2.0; // Avoid very small values
        }
        
        // Parse epsilon value
        double eps = 1e-6; // Default epsilon
        if (offset + sizeof(float) <= Size) {
            float eps_value;
            std::memcpy(&eps_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is non-negative
            eps = std::abs(eps_value);
        }
        
        // Parse keepdim flag
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1; // Use lowest bit
        }
        
        // Apply pairwise_distance operation
        torch::Tensor result = torch::pairwise_distance(x1, x2, p, eps, keepdim);
        
        // Ensure result is not empty
        if (result.numel() > 0) {
            volatile double sum = result.sum().item<double>();
            (void)sum; // Prevent optimization from removing the computation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

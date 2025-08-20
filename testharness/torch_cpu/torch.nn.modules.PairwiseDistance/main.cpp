#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for PairwiseDistance
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for PairwiseDistance
        double p = 2.0;  // Default p-norm value
        bool keepdim = false;
        double eps = 1e-6;
        
        // Extract p value from input data if available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure p is a reasonable value
            p = std::abs(p);
            if (p == 0.0) p = 2.0;  // Avoid division by zero
        }
        
        // Extract keepdim from input data if available
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;  // Use lowest bit to determine boolean
        }
        
        // Extract eps value from input data if available
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is non-negative
            eps = std::abs(eps);
        }
        
        // Create PairwiseDistance module
        torch::nn::PairwiseDistance pairwise_distance(
            torch::nn::PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim)
        );
        
        // Apply the operation
        torch::Tensor output = pairwise_distance->forward(x1, x2);
        
        // Try to access the output tensor to ensure computation is performed
        if (output.defined()) {
            auto sizes = output.sizes();
            auto numel = output.numel();
            
            // Try to access some values if tensor is not empty
            if (numel > 0) {
                auto first_val = output.item<float>();
                (void)first_val;  // Prevent unused variable warning
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
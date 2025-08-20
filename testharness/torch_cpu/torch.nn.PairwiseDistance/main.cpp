#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for pairwise distance
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if we have data left
        torch::Tensor x2;
        if (offset < Size) {
            x2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, create a tensor with same shape as x1
            x2 = torch::ones_like(x1);
        }
        
        // Extract parameters for PairwiseDistance from remaining data
        double p = 2.0; // Default p-norm value
        bool keepdim = false;
        double eps = 1e-6;
        
        if (offset + 1 < Size) {
            // Use remaining data to determine p value (1, 2, or other)
            uint8_t p_selector = Data[offset++];
            if (p_selector % 3 == 0) {
                p = 1.0;
            } else if (p_selector % 3 == 1) {
                p = 2.0;
            } else {
                // Use a different p value
                p = 0.5 + (p_selector % 10);
            }
        }
        
        if (offset < Size) {
            // Use remaining data to determine keepdim
            keepdim = (Data[offset++] % 2 == 0);
        }
        
        if (offset < Size) {
            // Use remaining data to determine eps
            uint8_t eps_selector = Data[offset++];
            eps = 1e-8 * (1 + eps_selector % 100);
        }
        
        // Try to make tensors compatible for pairwise distance
        // PairwiseDistance expects tensors with same shape except possibly the last dimension
        if (x1.dim() > 0 && x2.dim() > 0) {
            // Try to reshape tensors to have compatible shapes if needed
            if (x1.sizes() != x2.sizes()) {
                // For pairwise distance, all dimensions except the last should match
                std::vector<int64_t> new_shape1, new_shape2;
                
                if (x1.dim() >= 2 && x2.dim() >= 2) {
                    // Keep all dimensions except the last one the same
                    for (int i = 0; i < std::min(x1.dim(), x2.dim()) - 1; i++) {
                        int64_t common_dim = std::min(x1.size(i), x2.size(i));
                        new_shape1.push_back(common_dim);
                        new_shape2.push_back(common_dim);
                    }
                    
                    // Keep the last dimensions as they are
                    new_shape1.push_back(x1.size(-1));
                    new_shape2.push_back(x2.size(-1));
                    
                    // Try to reshape
                    try {
                        x1 = x1.reshape(new_shape1);
                        x2 = x2.reshape(new_shape2);
                    } catch (...) {
                        // If reshape fails, create new tensors with compatible shapes
                        x1 = torch::ones(new_shape1, x1.options());
                        x2 = torch::ones(new_shape2, x2.options());
                    }
                }
            }
        }
        
        // Create PairwiseDistance module
        torch::nn::PairwiseDistance pairwise_distance(
            torch::nn::PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim)
        );
        
        // Apply pairwise distance
        torch::Tensor output = pairwise_distance->forward(x1, x2);
        
        // Try alternative ways to compute pairwise distance
        if (offset < Size) {
            uint8_t alt_method = Data[offset++];
            if (alt_method % 3 == 0) {
                // Use functional interface
                output = torch::pairwise_distance(x1, x2, p, eps, keepdim);
            } else if (alt_method % 3 == 1 && p == 2.0) {
                // For p=2, try using pdist
                if (x1.dim() == 2) {
                    output = torch::pdist(x1, 2.0);
                }
            }
        }
        
        // Access output elements to ensure computation is performed
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            if (std::isnan(sum) || std::isinf(sum)) {
                // This is not an error, just a result of the computation
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
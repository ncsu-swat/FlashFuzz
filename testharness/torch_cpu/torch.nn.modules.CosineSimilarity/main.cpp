#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Create first input tensor for cosine similarity
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left
        if (offset >= Size) {
            return 0;
        }
        
        // Create second tensor with same shape as first to ensure compatibility
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension parameter from the input data
        int64_t dim = 1;  // Default dimension
        if (x1.dim() > 0 && offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % x1.dim();
        } else if (x1.dim() == 0) {
            dim = 0;
        }
        
        // Get eps parameter from the input data
        double eps = 1e-8;  // Default epsilon
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure eps is a valid positive value
            if (!std::isnan(eps_f) && !std::isinf(eps_f) && eps_f > 0) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Ensure both tensors are float type for cosine similarity
        x1 = x1.to(torch::kFloat32);
        x2 = x2.to(torch::kFloat32);
        
        // Try to make tensors have compatible shapes
        // CosineSimilarity requires tensors to be broadcastable
        try {
            // Attempt to broadcast x2 to x1's shape if possible
            if (x1.sizes() != x2.sizes()) {
                // Try to expand x2 to match x1
                try {
                    x2 = x2.expand_as(x1);
                } catch (...) {
                    // If expansion fails, try the other way
                    try {
                        x1 = x1.expand_as(x2);
                    } catch (...) {
                        // Create x2 with same shape as x1
                        x2 = torch::randn(x1.sizes());
                    }
                }
            }
        } catch (...) {
            // Fallback: create matching tensors
            x2 = torch::randn(x1.sizes());
        }
        
        // Ensure dim is valid for the tensor
        if (x1.dim() > 0) {
            dim = dim % x1.dim();
        } else {
            dim = 0;
        }
        
        // Create CosineSimilarity module with fuzzed parameters
        torch::nn::CosineSimilarity cosine_similarity(
            torch::nn::CosineSimilarityOptions().dim(dim).eps(eps));
        
        // Apply the operation
        torch::Tensor output = cosine_similarity->forward(x1, x2);
        
        // Try with different dimensions if possible
        if (x1.dim() > 1 && offset < Size) {
            int64_t new_dim = static_cast<int64_t>(Data[offset++]) % x1.dim();
            try {
                torch::nn::CosineSimilarity cosine_similarity2(
                    torch::nn::CosineSimilarityOptions().dim(new_dim).eps(eps));
                torch::Tensor output2 = cosine_similarity2->forward(x1, x2);
            } catch (...) {
                // Silently ignore expected failures for different dimensions
            }
        }
        
        // Try with negative dimension (PyTorch supports negative indexing)
        if (x1.dim() > 0 && offset < Size) {
            int64_t neg_dim = -(static_cast<int64_t>(Data[offset++]) % x1.dim()) - 1;
            try {
                torch::nn::CosineSimilarity cosine_similarity_neg(
                    torch::nn::CosineSimilarityOptions().dim(neg_dim).eps(eps));
                torch::Tensor output_neg = cosine_similarity_neg->forward(x1, x2);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Try with a very small epsilon
        try {
            double small_eps = 1e-20;
            torch::nn::CosineSimilarity cosine_similarity_small_eps(
                torch::nn::CosineSimilarityOptions().dim(dim).eps(small_eps));
            torch::Tensor output_small_eps = cosine_similarity_small_eps->forward(x1, x2);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Try with a large epsilon
        try {
            double large_eps = 1.0;
            torch::nn::CosineSimilarity cosine_similarity_large_eps(
                torch::nn::CosineSimilarityOptions().dim(dim).eps(large_eps));
            torch::Tensor output_large_eps = cosine_similarity_large_eps->forward(x1, x2);
        } catch (...) {
            // Silently ignore expected failures
        }
        
        // Try with zero tensors
        try {
            torch::Tensor zero_tensor = torch::zeros_like(x1);
            torch::Tensor output_zero = cosine_similarity->forward(zero_tensor, x2);
        } catch (...) {
            // Silently ignore - division by near-zero norm is expected to be handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // keep the input
}
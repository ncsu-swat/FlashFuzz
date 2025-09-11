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
        
        // Need at least a few bytes for meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for cosine similarity
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension parameter from the input data
        int64_t dim = 1;  // Default dimension
        if (offset < Size) {
            // Use the next byte to determine the dimension
            dim = static_cast<int64_t>(Data[offset++]) % (std::max(1, static_cast<int>(x1.dim())));
        }
        
        // Get eps parameter from the input data
        double eps = 1e-8;  // Default epsilon
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) {
                eps = 1e-8;
            }
        }
        
        // Create CosineSimilarity module with different parameters
        torch::nn::CosineSimilarity cosine_similarity(torch::nn::CosineSimilarityOptions().dim(dim).eps(eps));
        
        // Apply the operation
        torch::Tensor output = cosine_similarity->forward(x1, x2);
        
        // Try with different dimensions if possible
        if (x1.dim() > 1 && offset < Size) {
            int64_t new_dim = static_cast<int64_t>(Data[offset++]) % x1.dim();
            torch::nn::CosineSimilarity cosine_similarity2(torch::nn::CosineSimilarityOptions().dim(new_dim).eps(eps));
            torch::Tensor output2 = cosine_similarity2->forward(x1, x2);
        }
        
        // Try with a very small epsilon
        if (offset < Size) {
            double small_eps = 1e-20;
            torch::nn::CosineSimilarity cosine_similarity_small_eps(torch::nn::CosineSimilarityOptions().dim(dim).eps(small_eps));
            torch::Tensor output_small_eps = cosine_similarity_small_eps->forward(x1, x2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

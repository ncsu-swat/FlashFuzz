#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
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
            int8_t dim_raw;
            std::memcpy(&dim_raw, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // Allow negative dimensions to test edge cases
            dim = static_cast<int64_t>(dim_raw);
        }
        
        // Get eps parameter from the input data
        double eps = 1e-8;  // Default epsilon
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create CosineSimilarity module with the dimension and eps
        torch::nn::CosineSimilarity cosine_similarity(torch::nn::CosineSimilarityOptions().dim(dim).eps(eps));
        
        // Apply the cosine similarity operation
        torch::Tensor output = cosine_similarity->forward(x1, x2);
        
        // Perform some operations on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum();
            if (sum.defined() && !sum.isnan().item<bool>() && !sum.isinf().item<bool>()) {
                volatile double result = sum.item<double>();
                (void)result;
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
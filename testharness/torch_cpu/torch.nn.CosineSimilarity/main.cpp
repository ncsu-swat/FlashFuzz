#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 10) {
            return 0;
        }
        
        // Create first input tensor for cosine similarity
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        // Create second tensor with same shape as first for compatibility
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make shapes compatible if they differ
        if (x1.sizes() != x2.sizes()) {
            // Reshape x2 to match x1's shape if possible
            try {
                int64_t x1_numel = x1.numel();
                int64_t x2_numel = x2.numel();
                if (x1_numel > 0 && x2_numel > 0) {
                    // Use the smaller tensor and expand/repeat to match
                    if (x2_numel >= x1_numel) {
                        x2 = x2.flatten().slice(0, 0, x1_numel).reshape(x1.sizes());
                    } else {
                        x1 = x1.flatten().slice(0, 0, x2_numel).reshape(x2.sizes());
                    }
                }
            } catch (...) {
                // If reshaping fails, create x2 with same shape as x1
                x2 = torch::randn(x1.sizes());
            }
        }
        
        // Ensure tensors have at least the dimension we'll operate on
        if (x1.dim() == 0 || x2.dim() == 0) {
            return 0;
        }
        
        // Get dimension parameter from the input data
        int64_t dim = 1;  // Default dimension
        if (offset < Size) {
            // Use the next byte to determine the dimension
            int8_t dim_raw;
            std::memcpy(&dim_raw, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // Clamp dimension to valid range
            int64_t ndim = x1.dim();
            dim = static_cast<int64_t>(dim_raw) % ndim;
            if (dim < 0) {
                dim += ndim;
            }
        } else {
            // Use dimension 0 if tensor is 1D, else 1
            dim = (x1.dim() > 1) ? 1 : 0;
        }
        
        // Get eps parameter from the input data - use a safe approach
        double eps = 1e-8;  // Default epsilon
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to a reasonable eps range [1e-12, 1e-1]
            eps = 1e-12 + (static_cast<double>(eps_raw) / static_cast<double>(UINT32_MAX)) * (1e-1 - 1e-12);
        }
        
        // Create CosineSimilarity module with the dimension and eps
        torch::nn::CosineSimilarity cosine_similarity(
            torch::nn::CosineSimilarityOptions().dim(dim).eps(eps)
        );
        
        // Apply the cosine similarity operation
        torch::Tensor output = cosine_similarity->forward(x1, x2);
        
        // Perform some operations on the output to ensure it's used
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            if (sum.defined()) {
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
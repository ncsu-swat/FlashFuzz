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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for LocalResponseNorm from the remaining data
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse size parameter (must be positive)
        int64_t size_raw;
        std::memcpy(&size_raw, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        int64_t size = std::abs(size_raw) % 10 + 1; // Ensure positive and reasonable size
        
        // Parse alpha parameter
        double alpha = 1e-4;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse beta parameter
        double beta = 0.75;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse k parameter
        double k = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&k, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create LocalResponseNorm module
        torch::nn::LocalResponseNorm lrn(
            torch::nn::LocalResponseNormOptions(size)
                .alpha(alpha)
                .beta(beta)
                .k(k)
        );
        
        // Apply LocalResponseNorm to the input tensor
        torch::Tensor output = lrn->forward(input);
        
        // Ensure the output is valid by performing a simple operation
        torch::Tensor sum = output.sum();
        
        // Test with different input shapes
        if (offset + 2 < Size && input.dim() > 0) {
            // Create a new tensor with different shape but same data type
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try applying LRN to this tensor too
            try {
                torch::Tensor output2 = lrn->forward(input2);
                torch::Tensor sum2 = output2.sum();
            } catch (const std::exception&) {
                // Silently catch exceptions for the second tensor
            }
        }
        
        // Test with edge case parameters
        try {
            // Create LRN with extreme parameters
            torch::nn::LocalResponseNorm lrn_extreme(
                torch::nn::LocalResponseNormOptions(1)
                    .alpha(1e-10)
                    .beta(1e-10)
                    .k(1e-10)
            );
            torch::Tensor output_extreme = lrn_extreme->forward(input);
        } catch (const std::exception&) {
            // Silently catch exceptions for extreme parameters
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

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
        
        // Skip if we don't have enough data
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract alpha parameter from the remaining data if available
        double alpha = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure alpha is not zero or negative (which would cause issues)
            if (alpha <= 0.0) {
                alpha = 1.0;
            }
        }
        
        // Make a copy of the input tensor to preserve original data
        torch::Tensor original = input.clone();
        
        // Apply celu_ in-place operation using torch::celu_
        torch::celu_(input, alpha);
        
        // Verify the operation by comparing with the non-in-place version
        torch::Tensor expected = torch::celu(original, alpha);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        if (!torch::allclose(input, expected)) {
            throw std::runtime_error("In-place celu_ produced different result than non-in-place celu");
        }
        
        // Test edge cases with new tensors if we have more data
        if (offset < Size) {
            // Create another tensor for additional testing
            torch::Tensor edge_case = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test with different alpha values
            double edge_alpha = 0.5;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&edge_alpha, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure alpha is positive but allow very small values to test edge cases
                if (edge_alpha <= 0.0) {
                    edge_alpha = std::numeric_limits<double>::min();
                }
            }
            
            // Apply celu_ in-place using torch::celu_
            torch::celu_(edge_case, edge_alpha);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for bincount
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bincount requires integer input, so convert if needed
        if (input.dtype() != torch::kInt64 && input.dtype() != torch::kInt32 && input.dtype() != torch::kInt16 && input.dtype() != torch::kInt8) {
            input = input.to(torch::kInt64);
        }
        
        // Create optional weights tensor
        bool use_weights = false;
        torch::Tensor weights;
        
        // Use remaining data to decide if we should use weights
        if (offset < Size) {
            use_weights = (Data[offset++] % 2 == 0);
            
            if (use_weights && offset < Size) {
                weights = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weights has same size as input
                if (weights.sizes() != input.sizes()) {
                    weights = weights.expand_as(input);
                }
            }
        }
        
        // Get minlength parameter (optional)
        int64_t minlength = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&minlength, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure minlength is not too large to avoid excessive memory usage
            minlength = std::abs(minlength) % 1000;
        }
        
        // Call bincount with different parameter combinations
        torch::Tensor result;
        
        if (use_weights && !weights.defined()) {
            // If we wanted to use weights but couldn't create a valid one
            result = torch::bincount(input, {}, minlength);
        } 
        else if (use_weights) {
            // Convert weights to float if needed
            if (weights.dtype() != torch::kFloat && weights.dtype() != torch::kDouble) {
                weights = weights.to(torch::kFloat);
            }
            result = torch::bincount(input, weights, minlength);
        } 
        else {
            // No weights
            result = torch::bincount(input, {}, minlength);
        }
        
        // Access result to ensure computation is performed
        auto result_size = result.sizes();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
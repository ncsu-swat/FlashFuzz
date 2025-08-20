#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the round_ operation in-place
        tensor.round_();
        
        // Verify the operation worked by comparing with non-inplace version
        torch::Tensor expected = torch::round(original);
        
        // Check if the results match
        if (!torch::allclose(tensor, expected)) {
            std::cerr << "Inplace and out-of-place round operations produced different results" << std::endl;
        }
        
        // Try with decimals parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t decimals;
            std::memcpy(&decimals, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Create a new tensor for this test
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor original2 = tensor2.clone();
            
            // Apply round_ with decimals parameter
            tensor2.round_(decimals);
            
            // Verify with non-inplace version
            torch::Tensor expected2 = torch::round(original2, decimals);
            
            if (!torch::allclose(tensor2, expected2)) {
                std::cerr << "Inplace and out-of-place round operations with decimals produced different results" << std::endl;
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
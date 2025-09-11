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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the original tensor for comparison
        torch::Tensor original = tensor.clone();
        
        // Apply the erf_ operation in-place
        tensor.erf_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected = torch::erf(original);
        
        // Check if the operation was applied correctly
        if (tensor.sizes() != expected.sizes() || 
            tensor.dtype() != expected.dtype() ||
            !torch::allclose(tensor, expected, 1e-5, 1e-8)) {
            throw std::runtime_error("erf_ operation produced unexpected results");
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

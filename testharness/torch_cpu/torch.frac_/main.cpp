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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor to verify the in-place operation
        torch::Tensor original = input_tensor.clone();
        
        // Apply the frac_ operation (in-place)
        input_tensor.frac_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        // The frac operation returns the fractional portion of each element
        // For each element x, frac(x) = x - floor(x)
        torch::Tensor expected = original - original.floor();
        
        // Check if the operation produced the expected result
        // This is not necessary for fuzzing but helps verify correctness
        if (input_tensor.numel() > 0 && !torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
            // This shouldn't happen unless there's a bug in PyTorch
            throw std::runtime_error("frac_ operation produced unexpected results");
        }
        
        // Try another variant with a different tensor
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.frac_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
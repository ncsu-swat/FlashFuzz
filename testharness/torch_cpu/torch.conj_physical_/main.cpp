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
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply conj_physical_ operation (in-place conjugation)
        tensor.conj_physical_();
        
        // Verify the operation worked correctly for complex tensors
        if (tensor.is_complex()) {
            // For complex tensors, the real part should remain the same
            // and the imaginary part should be negated
            torch::Tensor expected = torch::complex(torch::real(original), -torch::imag(original));
            
            // Check if the result matches the expected output
            bool equal = torch::allclose(tensor, expected);
            if (!equal) {
                throw std::runtime_error("conj_physical_ operation did not produce expected result");
            }
        } else {
            // For non-complex tensors, the tensor should remain unchanged
            bool equal = torch::allclose(tensor, original);
            if (!equal) {
                throw std::runtime_error("conj_physical_ modified a non-complex tensor");
            }
        }
        
        // Try to create another tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            tensor2.conj_physical_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

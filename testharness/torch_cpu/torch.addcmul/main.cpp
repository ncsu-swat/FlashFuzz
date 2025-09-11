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
        
        // Need at least 3 tensors for addcmul: input, tensor1, tensor2
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
            
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create tensor1
        torch::Tensor tensor1;
        if (offset < Size) {
            tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor1 = torch::ones_like(input);
        }
        
        // Create tensor2
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = torch::ones_like(input);
        }
        
        // Parse value for alpha
        double alpha = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply addcmul operation
        torch::Tensor result = torch::addcmul(input, tensor1, tensor2, alpha);
        
        // Try in-place version if possible
        if (input.is_floating_point() && input.sizes() == result.sizes()) {
            torch::Tensor input_copy = input.clone();
            input_copy.addcmul_(tensor1, tensor2, alpha);
        }
        
        // Try the functional version with different alpha values
        if (offset + sizeof(double) <= Size) {
            double alpha2;
            std::memcpy(&alpha2, Data + offset, sizeof(double));
            torch::Tensor result2 = torch::addcmul(input, tensor1, tensor2, alpha2);
        }
        
        // Try with scalar inputs
        if (offset < Size) {
            double scalar_value;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Test with scalar as tensor1
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor scalar_result2 = torch::addcmul(input, scalar_tensor, tensor2, alpha);
                
                // Test with scalar as tensor2
                torch::Tensor scalar_result3 = torch::addcmul(input, tensor1, scalar_tensor, alpha);
            }
        }
        
        // Try with empty tensors if we have enough data
        if (offset < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            try {
                torch::Tensor empty_result = torch::addcmul(empty_tensor, tensor1, tensor2, alpha);
            } catch (...) {
                // Expected to potentially fail, but let's try it
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

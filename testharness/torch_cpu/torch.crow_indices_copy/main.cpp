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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create the sparse tensor
        torch::Tensor crow_indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the crow_indices_copy operation
        torch::Tensor result = torch::crow_indices_copy(crow_indices);
        
        // Try different variants of the API
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Variant 1: Just the crow_indices
            if (variant % 3 == 0) {
                result = torch::crow_indices_copy(crow_indices);
            }
            // Variant 2: Create another tensor and copy its crow_indices
            else if (variant % 3 == 1 && offset < Size) {
                torch::Tensor other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                result = torch::crow_indices_copy(other_tensor);
            }
            // Variant 3: Try with different tensor properties
            else if (offset < Size) {
                torch::Tensor alt_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                result = torch::crow_indices_copy(alt_tensor);
            }
        }
        
        // Access the result to ensure the operation is executed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            auto dtype = result.dtype();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

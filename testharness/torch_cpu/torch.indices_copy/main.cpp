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
        
        // Create source tensor
        torch::Tensor source = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the indices_copy operation (only takes one argument)
        torch::Tensor result = torch::indices_copy(source);
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum();
        
        // Try with different tensor
        if (offset < Size) {
            torch::Tensor alt_source = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Try the operation again
            try {
                result = torch::indices_copy(alt_source);
                sum += result.sum();
            } catch (const std::exception&) {
                // Ignore exceptions from this second attempt
            }
        }
        
        // Try with cloned tensor
        if (offset < Size) {
            torch::Tensor cloned_source = source.clone();
            try {
                result = torch::indices_copy(cloned_source);
                sum += result.sum();
            } catch (const std::exception&) {
                // Ignore exceptions
            }
        }
        
        // Try with different data types
        if (offset < Size) {
            try {
                torch::Tensor float_source = source.to(torch::kFloat32);
                result = torch::indices_copy(float_source);
                sum += result.sum();
            } catch (const std::exception&) {
                // Ignore exceptions
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
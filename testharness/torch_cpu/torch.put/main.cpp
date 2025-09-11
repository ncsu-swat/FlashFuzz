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
        
        // Create the source tensor
        torch::Tensor source = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the destination tensor
        torch::Tensor destination = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create index tensor
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to convert to long for indices if needed
            if (index.scalar_type() != torch::kLong) {
                try {
                    index = index.to(torch::kLong);
                } catch (...) {
                    // If conversion fails, create a simple index tensor
                    index = torch::tensor({0}, torch::kLong);
                }
            }
        } else {
            index = torch::tensor({0}, torch::kLong);
        }
        
        // Create values tensor
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Default values if we've exhausted the input data
            values = torch::ones_like(source);
        }
        
        // Apply torch.put operation
        try {
            // Variant 1: Using put_ directly on the tensor
            torch::Tensor result1 = destination.clone();
            result1.put_(index, values);
            
            // Variant 2: Using functional API
            torch::Tensor result2 = destination.clone();
            torch::put(result2, index, values);
            
            // Variant 3: Using accumulate option
            bool accumulate = false;
            if (offset < Size) {
                accumulate = Data[offset++] % 2 == 1;
            }
            
            torch::Tensor result3 = destination.clone();
            result3.put_(index, values, accumulate);
            
            torch::Tensor result4 = destination.clone();
            torch::put(result4, index, values, accumulate);
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (like dimension mismatch) are fine
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

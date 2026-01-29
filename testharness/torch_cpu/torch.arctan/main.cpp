#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for arctan operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arctan operation
        torch::Tensor result = torch::arctan(input);
        
        // Try in-place version if there's more data
        if (offset < Size && Size - offset > 0) {
            torch::Tensor input_copy = input.clone();
            input_copy.arctan_();
        }
        
        // Try arctan with output argument using compatible dtype
        if (offset < Size && Size - offset > 0) {
            // Create output tensor with same dtype as would be returned by arctan
            torch::Tensor output = torch::empty_like(result);
            
            try {
                // Try arctan with output argument
                torch::arctan_out(output, input);
            } catch (...) {
                // Silently ignore dtype/shape mismatch errors
            }
        }
        
        // Try arctan2 if we have more data to create a second tensor
        if (offset < Size && Size - offset > 2) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                // Apply arctan2 operation (atan2(input, input2))
                // arctan2 requires tensors that can be broadcast together
                torch::Tensor result2 = torch::arctan2(input, input2);
                
                // Try in-place version of arctan2
                if (offset < Size && Size - offset > 0) {
                    torch::Tensor input_copy = input.clone();
                    input_copy.arctan2_(input2);
                }
            } catch (...) {
                // Silently ignore shape/broadcasting errors for arctan2
            }
        }
        
        // Also test atan (alias for arctan)
        if (offset < Size) {
            torch::Tensor atan_result = torch::atan(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
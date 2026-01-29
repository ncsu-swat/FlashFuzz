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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply logical_not operation
        torch::Tensor result = torch::logical_not(input_tensor);
        
        // Try some variations if we have enough data
        if (offset + 1 < Size) {
            // Try in-place version on a clone
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.logical_not_();
            
            // Try with out parameter - create boolean output tensor
            torch::Tensor out_tensor = torch::empty_like(input_tensor, torch::kBool);
            torch::logical_not_out(out_tensor, input_tensor);
        }
        
        // Try with different tensor types to improve coverage
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply logical_not to this tensor too
            torch::Tensor another_result = torch::logical_not(another_tensor);
            
            // Test with boolean tensor explicitly
            torch::Tensor bool_tensor = input_tensor.to(torch::kBool);
            torch::Tensor bool_result = torch::logical_not(bool_tensor);
        }
        
        // Test with specific dtypes for better coverage
        if (Size > 4) {
            try {
                // Test with integer tensor
                torch::Tensor int_tensor = input_tensor.to(torch::kInt);
                torch::Tensor int_result = torch::logical_not(int_tensor);
                
                // Test with float tensor
                torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                torch::Tensor float_result = torch::logical_not(float_tensor);
            } catch (...) {
                // Silently handle dtype conversion failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Tanhshrink module
        torch::nn::Tanhshrink tanhshrink_module;
        
        // Apply Tanhshrink operation
        torch::Tensor output = tanhshrink_module->forward(input);
        
        // Alternative implementation to test against
        torch::Tensor expected_output = input - torch::tanh(input);
        
        // Compare results
        fuzzer_utils::compareTensors(output, expected_output, Data, Size);
        
        // Try with different tensor options
        if (offset + 1 < Size) {
            // Create a new tensor with different options
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply Tanhshrink operation again
            torch::Tensor output2 = tanhshrink_module->forward(input2);
            
            // Verify the alternative implementation
            torch::Tensor expected_output2 = input2 - torch::tanh(input2);
            fuzzer_utils::compareTensors(output2, expected_output2, Data, Size);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

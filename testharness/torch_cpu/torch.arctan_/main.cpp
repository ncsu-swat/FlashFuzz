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
        
        // Create input tensor and apply arctan_ in-place
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        input.arctan_();
        
        // Try to create additional tensors if there's more data
        // to exercise the API with various tensor configurations
        while (offset + 2 <= Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            another_input.arctan_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
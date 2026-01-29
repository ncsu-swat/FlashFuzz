#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes for tensor creation and dropout parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point (dropout requires float type)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract dropout probability from the input data
        double p = 0.5; // Default value
        if (offset < Size) {
            // Use a byte to generate probability in [0, 1]
            p = static_cast<double>(Data[offset++]) / 255.0;
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply dropout_ in-place operation using torch::dropout_
        // This is the main API we're testing
        torch::dropout_(input, p, train);
        
        // Basic sanity check - ensure output tensor is valid
        if (input.numel() > 0) {
            // Access the tensor to ensure it's in a valid state
            auto sum = input.sum();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
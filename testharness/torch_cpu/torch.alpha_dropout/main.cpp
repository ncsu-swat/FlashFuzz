#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::floor

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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (alpha_dropout works with float tensors)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is floating point (required for alpha_dropout)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract p (dropout probability) from the input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN and Inf
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            }
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p < 1
        }
        
        // Extract training flag from the input data
        bool training = true; // Default value
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Extract inplace flag from the input data
        bool inplace = false; // Default value
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Apply alpha_dropout using the functional API
        torch::Tensor output;
        
        try {
            if (inplace) {
                // Make a copy for inplace operation
                torch::Tensor input_copy = input.clone();
                output = torch::alpha_dropout_(input_copy, p, training);
            } else {
                output = torch::alpha_dropout(input, p, training);
            }
            
            // Verify output tensor is valid
            (void)output.sizes();
            (void)output.numel();
        } catch (const c10::Error &e) {
            // Expected errors from invalid tensor configurations
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
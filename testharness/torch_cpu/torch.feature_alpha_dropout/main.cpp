#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::floor, std::isnan, std::isinf

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // feature_alpha_dropout requires floating point tensors
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
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p < 1
        }
        
        // Extract training flag from the input data
        bool training = true;
        if (offset < Size) {
            training = Data[offset++] & 0x1; // Use lowest bit to determine training mode
        }
        
        // Extract inplace flag from the input data
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use lowest bit to determine inplace mode
        }
        
        // Apply feature_alpha_dropout
        torch::Tensor output;
        try {
            if (inplace) {
                // Make a copy for inplace operation to preserve original
                torch::Tensor input_copy = input.clone();
                torch::feature_alpha_dropout_(input_copy, p, training);
                output = input_copy;
            } else {
                output = torch::feature_alpha_dropout(input, p, training);
            }
            
            // Verify output tensor is valid
            (void)output.sizes();
            (void)output.numel();
        } catch (const c10::Error &e) {
            // Expected errors from invalid tensor configurations (e.g., wrong dimensions)
            return 0;
        }
        
        // Test with different p values if we have more data
        if (offset < Size) {
            try {
                // Test with no dropout (p=0)
                torch::Tensor out_p0 = torch::feature_alpha_dropout(input, 0.0, training);
                (void)out_p0.numel();
                
                // Test with high dropout (p close to 1, but not exactly 1 to avoid issues)
                torch::Tensor out_p1 = torch::feature_alpha_dropout(input, 0.9, training);
                (void)out_p1.numel();
            } catch (const c10::Error &e) {
                // Silently handle expected failures
            }
        }
        
        // Test with different training modes
        if (offset < Size) {
            try {
                torch::Tensor out_train = torch::feature_alpha_dropout(input, p, !training);
                (void)out_train.numel();
            } catch (const c10::Error &e) {
                // Silently handle expected failures
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
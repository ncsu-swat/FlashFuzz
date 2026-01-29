#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <torch/nn/functional/dropout.h>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Alpha dropout requires floating point tensor
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract probability parameter from the input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            } else {
                p = std::abs(p);
                p = p > 1.0f ? p - std::floor(p) : p;
            }
        }
        
        // Extract training parameter from the input data
        bool training = true; // Default value
        if (offset < Size) {
            training = Data[offset++] & 0x1;
        }
        
        // Create a copy of the input tensor for inplace operation
        torch::Tensor input_copy = input.clone();
        
        // Apply alpha_dropout with inplace=true (equivalent to alpha_dropout_)
        // Using the functional API which is the correct C++ interface
        namespace F = torch::nn::functional;
        
        try {
            auto result = F::alpha_dropout(input_copy, 
                F::AlphaDropoutFuncOptions().p(p).training(training).inplace(true));
        } catch (const c10::Error&) {
            // Silently catch expected errors (e.g., invalid tensor configurations)
        }
        
        // Also test non-inplace version for better coverage
        try {
            auto result_non_inplace = F::alpha_dropout(input,
                F::AlphaDropoutFuncOptions().p(p).training(training).inplace(false));
        } catch (const c10::Error&) {
            // Silently catch expected errors
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}
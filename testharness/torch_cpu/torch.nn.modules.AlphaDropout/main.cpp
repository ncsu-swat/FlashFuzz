#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for AlphaDropout
        float p = 0.5; // Default probability
        bool inplace = false;
        bool train = true;
        
        // Parse probability if we have more data
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure p is between 0 and 1
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part
            
            // Handle edge cases for NaN/Inf
            if (!std::isfinite(p)) {
                p = 0.5;
            }
        }
        
        // Parse inplace flag if we have more data
        if (offset < Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Parse train flag if we have more data
        if (offset < Size) {
            train = Data[offset++] & 0x1;
        }
        
        // Create AlphaDropout module
        torch::nn::AlphaDropout alpha_dropout(
            torch::nn::AlphaDropoutOptions().p(p).inplace(inplace)
        );
        
        // Set training mode
        if (train) {
            alpha_dropout->train();
        } else {
            alpha_dropout->eval();
        }
        
        // Apply AlphaDropout to the input tensor
        torch::Tensor output = alpha_dropout->forward(input_tensor);
        
        // Force computation to ensure any potential errors are triggered
        output.sum().item<float>();
        
        // Test with different batch sizes if tensor has at least 1 dimension
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            try {
                torch::Tensor first_element = input_tensor.slice(0, 0, 1);
                torch::Tensor output_first = alpha_dropout->forward(first_element);
                output_first.sum().item<float>();
            } catch (...) {
                // Silently ignore shape-related failures
            }
        }
        
        // Test with different training modes
        alpha_dropout->train(!train);
        torch::Tensor output2 = alpha_dropout->forward(input_tensor);
        output2.sum().item<float>();
        
        // Additional coverage: test with contiguous tensor
        if (!input_tensor.is_contiguous()) {
            torch::Tensor contiguous_input = input_tensor.contiguous();
            torch::Tensor output3 = alpha_dropout->forward(contiguous_input);
            output3.sum().item<float>();
        }
        
        // Test pretty_print for additional coverage
        alpha_dropout->pretty_print(std::cout);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
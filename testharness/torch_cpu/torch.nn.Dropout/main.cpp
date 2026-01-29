#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for basic operation
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        float p = 0.5f; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize the probability value to valid range [0, 1]
            if (std::isnan(p) || std::isinf(p)) {
                p = 0.5f;
            } else {
                // Clamp to valid range
                p = std::fmod(std::fabs(p), 1.0f);
            }
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Extract inplace flag from the input data
        bool inplace = false; // Default value
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create Dropout module with sanitized probability
        torch::nn::Dropout dropout_module(torch::nn::DropoutOptions().p(p).inplace(inplace));
        
        // Set the module to training or evaluation mode
        if (train) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        
        // Apply dropout to the input tensor
        // Need to clone if inplace to avoid modifying original
        torch::Tensor input_for_module = inplace ? input.clone() : input;
        torch::Tensor output = dropout_module->forward(input_for_module);
        
        // Test the functional interface as well
        try {
            torch::Tensor output_functional = torch::dropout(input.clone(), p, train);
        } catch (...) {
            // Silently ignore expected failures from functional interface
        }
        
        // Test with edge case probabilities if we have more data
        if (offset + sizeof(float) <= Size) {
            float edge_p;
            std::memcpy(&edge_p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize edge probability
            if (std::isnan(edge_p) || std::isinf(edge_p)) {
                edge_p = 0.0f;
            } else {
                edge_p = std::fmod(std::fabs(edge_p), 1.0f);
            }
            
            try {
                torch::nn::Dropout dropout_edge(torch::nn::DropoutOptions().p(edge_p).inplace(inplace));
                if (train) {
                    dropout_edge->train();
                } else {
                    dropout_edge->eval();
                }
                torch::Tensor input_for_edge = inplace ? input.clone() : input;
                torch::Tensor output_edge = dropout_edge->forward(input_for_edge);
            } catch (...) {
                // Silently ignore expected failures with edge values
            }
        }
        
        // Test boundary probabilities explicitly
        try {
            torch::nn::Dropout dropout_zero(torch::nn::DropoutOptions().p(0.0));
            dropout_zero->train();
            torch::Tensor out_zero = dropout_zero->forward(input.clone());
        } catch (...) {
            // Silently ignore
        }
        
        try {
            torch::nn::Dropout dropout_one(torch::nn::DropoutOptions().p(1.0));
            dropout_one->train();
            torch::Tensor out_one = dropout_one->forward(input.clone());
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
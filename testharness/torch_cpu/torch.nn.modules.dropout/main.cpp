#include "fuzzer_utils.h"
#include <iostream>

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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for dropout from the remaining data
        double p = 0.5;
        bool train = true;
        bool inplace = false;
        
        // Parse dropout probability if we have data left
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t p_byte = Data[offset++];
            // Map to [0, 1) range - p must be < 1 for valid dropout
            p = static_cast<double>(p_byte) / 256.0;
        }
        
        // Parse training mode flag if we have data left
        if (offset + sizeof(uint8_t) <= Size) {
            train = Data[offset++] & 0x1;
        }
        
        // Parse inplace flag if we have data left
        if (offset + sizeof(uint8_t) <= Size) {
            inplace = Data[offset++] & 0x1;
        }
        
        // Create dropout module
        torch::nn::Dropout dropout_module(torch::nn::DropoutOptions().p(p).inplace(inplace));
        
        // Set training mode
        if (train) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        
        // Apply dropout to the input tensor
        torch::Tensor output = dropout_module->forward(input);
        
        // Test functional interface as well
        torch::Tensor output2 = torch::dropout(input, p, train);
        
        // Test feature_dropout - requires specific dimensions, wrap in inner try-catch
        try {
            torch::Tensor output3 = torch::feature_dropout(input, p, train);
        } catch (...) {
            // Expected for invalid tensor shapes
        }
        
        // Test alpha_dropout
        torch::Tensor output4 = torch::alpha_dropout(input, p, train);
        
        // Test feature_alpha_dropout - requires specific dimensions
        try {
            torch::Tensor output5 = torch::feature_alpha_dropout(input, p, train);
        } catch (...) {
            // Expected for invalid tensor shapes
        }
        
        // Test AlphaDropout module
        torch::nn::AlphaDropout alpha_dropout_module(torch::nn::AlphaDropoutOptions().p(p).inplace(inplace));
        if (train) {
            alpha_dropout_module->train();
        } else {
            alpha_dropout_module->eval();
        }
        torch::Tensor output9 = alpha_dropout_module->forward(input);
        
        // Test Dropout2d module if input has at least 2 dimensions
        if (input.dim() >= 2) {
            try {
                torch::nn::Dropout2d dropout2d_module(torch::nn::Dropout2dOptions().p(p).inplace(inplace));
                if (train) {
                    dropout2d_module->train();
                } else {
                    dropout2d_module->eval();
                }
                torch::Tensor output11 = dropout2d_module->forward(input);
            } catch (...) {
                // Expected for some tensor configurations
            }
        }
        
        // Test Dropout3d module if input has at least 3 dimensions
        if (input.dim() >= 3) {
            try {
                torch::nn::Dropout3d dropout3d_module(torch::nn::Dropout3dOptions().p(p).inplace(inplace));
                if (train) {
                    dropout3d_module->train();
                } else {
                    dropout3d_module->eval();
                }
                torch::Tensor output12 = dropout3d_module->forward(input);
            } catch (...) {
                // Expected for some tensor configurations
            }
        }
        
        // Test functional dropout1d - note: Dropout1d module is not available in C++ frontend
        // but we can use the functional interface
        if (input.dim() >= 2 && input.dim() <= 3) {
            try {
                torch::Tensor output13 = torch::nn::functional::dropout1d(
                    input, 
                    torch::nn::functional::Dropout1dFuncOptions().p(p).inplace(false));
            } catch (...) {
                // Expected for some tensor configurations
            }
        }
        
        // Test FeatureAlphaDropout module
        try {
            torch::nn::FeatureAlphaDropout feature_alpha_module(torch::nn::FeatureAlphaDropoutOptions().p(p).inplace(inplace));
            if (train) {
                feature_alpha_module->train();
            } else {
                feature_alpha_module->eval();
            }
            torch::Tensor output14 = feature_alpha_module->forward(input);
        } catch (...) {
            // Expected for invalid tensor shapes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
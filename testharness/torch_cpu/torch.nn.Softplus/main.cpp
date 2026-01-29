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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse beta parameter from the remaining data
        double beta = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure beta is positive and not too extreme
            if (beta <= 0) {
                beta = 1.0;
            } else if (std::isnan(beta) || std::isinf(beta)) {
                beta = 1.0;
            } else if (beta > 1000.0) {
                beta = 1000.0;
            }
        }
        
        // Parse threshold parameter from the remaining data
        double threshold = 20.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure threshold is not too extreme
            if (std::isnan(threshold) || std::isinf(threshold)) {
                threshold = 20.0;
            } else if (threshold > 1000.0) {
                threshold = 1000.0;
            }
        }
        
        // Create Softplus module
        torch::nn::SoftplusOptions options;
        options.beta(beta);
        options.threshold(threshold);
        torch::nn::Softplus softplus_module(options);
        
        // Apply Softplus operation
        torch::Tensor output = softplus_module->forward(input);
        
        // Alternative way to apply Softplus using functional API
        torch::Tensor output_functional = torch::nn::functional::softplus(
            input, 
            torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(threshold)
        );
        
        // Try with different beta and threshold values
        if (offset + 2 * sizeof(double) <= Size) {
            double beta2, threshold2;
            std::memcpy(&beta2, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&threshold2, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Apply constraints to avoid invalid values
            if (beta2 > 0 && !std::isnan(beta2) && !std::isinf(beta2) && beta2 <= 1000.0 &&
                !std::isnan(threshold2) && !std::isinf(threshold2) && threshold2 <= 1000.0) {
                
                torch::nn::SoftplusOptions options2;
                options2.beta(beta2);
                options2.threshold(threshold2);
                torch::nn::Softplus softplus_module2(options2);
                torch::Tensor output2 = softplus_module2->forward(input);
            }
        }
        
        // Test with default parameters
        torch::nn::Softplus default_softplus;
        torch::Tensor default_output = default_softplus->forward(input);
        
        // Test with functional API and default parameters
        torch::Tensor default_output_functional = torch::nn::functional::softplus(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
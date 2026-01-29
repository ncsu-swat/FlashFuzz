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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse beta parameter from the input data
        double beta = 1.0;
        if (offset + sizeof(float) <= Size) {
            float beta_raw;
            std::memcpy(&beta_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure beta is positive and reasonable
            if (std::isfinite(beta_raw) && beta_raw > 0 && beta_raw < 1e6) {
                beta = static_cast<double>(beta_raw);
            }
        }
        
        // Parse threshold parameter from the input data
        double threshold = 20.0;
        if (offset + sizeof(float) <= Size) {
            float threshold_raw;
            std::memcpy(&threshold_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure threshold is positive and reasonable
            if (std::isfinite(threshold_raw) && threshold_raw > 0 && threshold_raw < 1e6) {
                threshold = static_cast<double>(threshold_raw);
            }
        }
        
        // Create Softplus module with options
        auto options = torch::nn::SoftplusOptions().beta(beta).threshold(threshold);
        torch::nn::Softplus softplus_module(options);
        
        // Apply Softplus operation using the module
        torch::Tensor output = softplus_module->forward(input);
        
        // Verify output is defined
        if (output.defined()) {
            auto sizes = output.sizes();
            (void)sizes;
        }
        
        // Try the functional version as well
        auto func_options = torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(threshold);
        torch::Tensor output_functional = torch::nn::functional::softplus(input, func_options);
        
        // Try with default parameters
        torch::Tensor output_default = torch::nn::functional::softplus(input);
        
        // Try with different parameter combinations based on fuzzer input
        if (offset + 1 <= Size) {
            uint8_t selector = Data[offset++];
            
            try {
                if (selector % 4 == 0) {
                    // Large beta value
                    auto opts = torch::nn::functional::SoftplusFuncOptions().beta(100.0).threshold(threshold);
                    torch::Tensor result = torch::nn::functional::softplus(input, opts);
                    (void)result;
                } else if (selector % 4 == 1) {
                    // Small beta value
                    auto opts = torch::nn::functional::SoftplusFuncOptions().beta(0.01).threshold(threshold);
                    torch::Tensor result = torch::nn::functional::softplus(input, opts);
                    (void)result;
                } else if (selector % 4 == 2) {
                    // Large threshold value
                    auto opts = torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(100.0);
                    torch::Tensor result = torch::nn::functional::softplus(input, opts);
                    (void)result;
                } else {
                    // Small threshold value
                    auto opts = torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(1.0);
                    torch::Tensor result = torch::nn::functional::softplus(input, opts);
                    (void)result;
                }
            } catch (...) {
                // Silently ignore expected failures from extreme values
            }
        }
        
        // Test with different input types
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            
            try {
                torch::Tensor float_input;
                if (dtype_selector % 3 == 0) {
                    float_input = input.to(torch::kFloat32);
                } else if (dtype_selector % 3 == 1) {
                    float_input = input.to(torch::kFloat64);
                } else {
                    float_input = input.to(torch::kFloat16);
                }
                
                torch::Tensor result = torch::nn::functional::softplus(float_input);
                (void)result;
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
        
        // Test inplace-like behavior by creating a new module instance
        if (offset + 1 <= Size) {
            double new_beta = 1.0 + (Data[offset++] % 10);
            auto new_options = torch::nn::SoftplusOptions().beta(new_beta).threshold(threshold);
            torch::nn::Softplus new_module(new_options);
            torch::Tensor new_output = new_module->forward(input);
            (void)new_output;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
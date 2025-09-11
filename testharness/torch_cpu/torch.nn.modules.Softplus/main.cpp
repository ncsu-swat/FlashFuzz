#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
            if (std::isfinite(beta_raw) && beta_raw > 0) {
                beta = beta_raw;
            }
        }
        
        // Parse threshold parameter from the input data
        double threshold = 20.0;
        if (offset + sizeof(float) <= Size) {
            float threshold_raw;
            std::memcpy(&threshold_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure threshold is positive and reasonable
            if (std::isfinite(threshold_raw) && threshold_raw > 0) {
                threshold = threshold_raw;
            }
        }
        
        // Create Softplus module
        torch::nn::Softplus softplus_module(torch::nn::SoftplusOptions().beta(beta).threshold(threshold));
        
        // Apply Softplus operation
        torch::Tensor output = softplus_module->forward(input);
        
        // Try to access the output tensor to ensure computation is complete
        if (output.defined()) {
            auto sizes = output.sizes();
            auto dtype = output.dtype();
        }
        
        // Try the functional version as well
        torch::Tensor output_functional = torch::nn::functional::softplus(input, torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(threshold));
        
        // Try with different beta and threshold values
        torch::Tensor output2 = torch::nn::functional::softplus(input);
        
        // Try with extreme values
        if (offset + 1 < Size) {
            uint8_t extreme_selector = Data[offset++];
            if (extreme_selector % 4 == 0) {
                // Very large beta
                torch::Tensor output_large_beta = torch::nn::functional::softplus(input, torch::nn::functional::SoftplusFuncOptions().beta(1e10).threshold(threshold));
            } else if (extreme_selector % 4 == 1) {
                // Very small beta
                torch::Tensor output_small_beta = torch::nn::functional::softplus(input, torch::nn::functional::SoftplusFuncOptions().beta(1e-10).threshold(threshold));
            } else if (extreme_selector % 4 == 2) {
                // Very large threshold
                torch::Tensor output_large_threshold = torch::nn::functional::softplus(input, torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(1e10));
            } else {
                // Very small threshold
                torch::Tensor output_small_threshold = torch::nn::functional::softplus(input, torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(1e-10));
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter from the remaining data
        double alpha = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse inplace parameter (0 = false, non-zero = true)
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] > 0;
        }
        
        // Create ELU module with the parsed alpha
        torch::nn::ELU elu_module(torch::nn::ELUOptions().alpha(alpha).inplace(inplace));
        
        // Apply ELU operation
        torch::Tensor output = elu_module->forward(input);
        
        // Verify output is valid
        if (output.numel() > 0) {
            output.sum().item<float>();
        }
        
        // Try the functional version as well
        torch::Tensor output2 = torch::nn::functional::elu(input, torch::nn::functional::ELUFuncOptions().alpha(alpha).inplace(false));
        
        // Verify output is valid
        if (output2.numel() > 0) {
            output2.sum().item<float>();
        }
        
        // Try with extreme alpha values if we have more data
        if (offset < Size) {
            // Use the next byte to determine an extreme alpha value
            uint8_t alpha_selector = Data[offset++];
            
            // Choose between very small, very large, zero, or negative alpha
            double extreme_alpha;
            switch (alpha_selector % 4) {
                case 0: extreme_alpha = 1e-10; break;
                case 1: extreme_alpha = 1e10; break;
                case 2: extreme_alpha = 0.0; break;
                case 3: extreme_alpha = -1.0; break;
            }
            
            // Create ELU with extreme alpha
            torch::nn::ELU extreme_elu(torch::nn::ELUOptions().alpha(extreme_alpha));
            torch::Tensor extreme_output = extreme_elu->forward(input);
            
            // Verify output is valid
            if (extreme_output.numel() > 0) {
                extreme_output.sum().item<float>();
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
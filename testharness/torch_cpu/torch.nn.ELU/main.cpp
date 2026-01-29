#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Parse alpha parameter from remaining data
        double alpha = 1.0; // Default value
        if (offset + sizeof(float) <= Size) {
            float alpha_f;
            std::memcpy(&alpha_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize alpha to avoid NaN/Inf issues
            if (std::isfinite(alpha_f)) {
                // Clamp to reasonable range
                alpha = std::max(-100.0, std::min(100.0, static_cast<double>(alpha_f)));
            }
        }
        
        // Parse inplace parameter (0 = false, non-zero = true)
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] & 1) != 0;
        }
        
        // Clone input for inplace operations to avoid affecting other tests
        torch::Tensor input_for_inplace = input.clone();
        
        // Create ELU module with the parsed alpha
        torch::nn::ELU elu_module(torch::nn::ELUOptions().alpha(alpha).inplace(inplace));
        
        // Apply ELU operation
        torch::Tensor output;
        if (inplace) {
            output = elu_module->forward(input_for_inplace);
        } else {
            output = elu_module->forward(input);
        }
        
        // Verify output is valid
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
        
        // Try the functional version as well (never inplace to preserve input)
        torch::Tensor output2 = torch::nn::functional::elu(
            input, 
            torch::nn::functional::ELUFuncOptions().alpha(alpha).inplace(false)
        );
        
        // Verify output is valid
        if (output2.numel() > 0) {
            volatile float sum = output2.sum().item<float>();
            (void)sum;
        }
        
        // Try with extreme alpha values if we have more data
        if (offset < Size) {
            uint8_t alpha_selector = Data[offset++];
            
            // Choose between various alpha values
            double extreme_alpha;
            switch (alpha_selector % 5) {
                case 0: extreme_alpha = 1e-6; break;
                case 1: extreme_alpha = 100.0; break;
                case 2: extreme_alpha = 0.0; break;
                case 3: extreme_alpha = -1.0; break;
                case 4: extreme_alpha = 0.5; break;
            }
            
            // Create ELU with extreme alpha (never inplace)
            torch::nn::ELU extreme_elu(torch::nn::ELUOptions().alpha(extreme_alpha).inplace(false));
            torch::Tensor extreme_output = extreme_elu->forward(input);
            
            // Verify output is valid
            if (extreme_output.numel() > 0) {
                volatile float sum = extreme_output.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test with different tensor dtypes if we have data
        if (offset < Size) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::nn::ELU float_elu(torch::nn::ELUOptions().alpha(alpha).inplace(false));
                torch::Tensor float_output = float_elu->forward(float_input);
                if (float_output.numel() > 0) {
                    volatile float sum = float_output.sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
            
            try {
                torch::Tensor double_input = input.to(torch::kFloat64);
                torch::nn::ELU double_elu(torch::nn::ELUOptions().alpha(alpha).inplace(false));
                torch::Tensor double_output = double_elu->forward(double_input);
                if (double_output.numel() > 0) {
                    volatile double sum = double_output.sum().item<double>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore dtype conversion issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter from the remaining data
        float alpha = 1.0; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure alpha is not NaN or infinity
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 1.0;
            }
            
            // Ensure alpha is positive (CELU requires positive alpha)
            if (alpha <= 0) {
                alpha = std::abs(alpha);
                if (alpha == 0) alpha = 1.0;
            }
        }
        
        // Parse inplace flag
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) == 1;
        }
        
        // Create CELU module with the parsed alpha
        torch::nn::CELU celu_module(torch::nn::CELUOptions().alpha(alpha).inplace(inplace));
        
        // Apply CELU to the input tensor
        torch::Tensor output = celu_module->forward(input);
        
        // Verify the output is valid
        if (output.numel() > 0) {
            output.sum().item<float>();
        }
        
        // Alternative approach: use the functional interface
        torch::Tensor output2 = torch::celu(input, alpha);
        
        // Verify the output is valid
        if (output2.numel() > 0) {
            output2.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

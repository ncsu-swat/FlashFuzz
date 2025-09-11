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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Dropout3d
        float p = 0.5; // Default dropout probability
        bool inplace = false;
        
        // If we have more data, use it for parameters
        if (offset + 2 <= Size) {
            // Extract p value (between 0 and 1)
            p = static_cast<float>(Data[offset]) / 255.0f;
            offset++;
            
            // Extract inplace flag
            inplace = Data[offset] % 2 == 1;
            offset++;
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout_module(torch::nn::Dropout3dOptions().p(p).inplace(inplace));
        
        // Set training mode based on remaining data if available
        bool training_mode = true;
        if (offset < Size) {
            training_mode = Data[offset] % 2 == 1;
            offset++;
        }
        
        // Set the module's training mode
        dropout_module->train(training_mode);
        
        // Apply Dropout3d to the input tensor
        torch::Tensor output = dropout_module->forward(input);
        
        // Verify output is not empty
        if (output.numel() > 0) {
            // Access some values to ensure computation happened
            if (output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
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

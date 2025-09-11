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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a parameter for weight initialization
        float weight_init = 0.25;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&weight_init, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Get a boolean to determine if we should use num_parameters > 1
        bool use_channel_wise = false;
        if (offset < Size) {
            use_channel_wise = Data[offset++] & 0x1;
        }
        
        // Create PReLU module
        torch::nn::PReLU prelu;
        
        // Configure PReLU parameters
        if (use_channel_wise && input.dim() > 1) {
            // Channel-wise PReLU (num_parameters = number of channels)
            int64_t num_channels = input.size(1);
            if (num_channels > 0) {
                prelu = torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(num_channels));
                
                // Initialize weights with values derived from input data
                torch::Tensor weights = torch::ones(num_channels) * weight_init;
                prelu->weight = weights;
            }
        } else {
            // Single parameter for all channels
            prelu = torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(1));
            prelu->weight = torch::ones(1) * weight_init;
        }
        
        // Apply PReLU to input tensor
        torch::Tensor output = prelu->forward(input);
        
        // Verify output has same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("PReLU output shape doesn't match input shape");
        }
        
        // Test with eval mode
        prelu->eval();
        torch::Tensor output_eval = prelu->forward(input);
        
        // Test with train mode
        prelu->train();
        torch::Tensor output_train = prelu->forward(input);
        
        // Test with zero weights
        prelu->weight.zero_();
        torch::Tensor output_zero = prelu->forward(input);
        
        // Test with negative weights
        prelu->weight = -torch::ones_like(prelu->weight);
        torch::Tensor output_neg = prelu->forward(input);
        
        // Test with extreme weights
        if (offset + sizeof(float) <= Size) {
            float extreme_value;
            std::memcpy(&extreme_value, Data + offset, sizeof(float));
            prelu->weight = torch::ones_like(prelu->weight) * extreme_value;
            torch::Tensor output_extreme = prelu->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

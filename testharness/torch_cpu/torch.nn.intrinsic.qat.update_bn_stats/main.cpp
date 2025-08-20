#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create running_mean and running_var tensors
        torch::Tensor running_mean;
        torch::Tensor running_var;
        
        // If we have enough data, create running_mean and running_var
        if (offset + 2 < Size) {
            running_mean = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset + 2 < Size) {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create default running_var if not enough data
                running_var = torch::ones_like(running_mean);
            }
        } else {
            // Create default running_mean and running_var
            if (input_tensor.dim() > 0) {
                int64_t num_features = input_tensor.size(1);
                running_mean = torch::zeros({num_features});
                running_var = torch::ones({num_features});
            } else {
                running_mean = torch::zeros({1});
                running_var = torch::ones({1});
            }
        }
        
        // Get momentum value from the input data
        float momentum = 0.1f; // Default value
        if (offset < Size) {
            // Use the next byte to generate a momentum value between 0 and 1
            momentum = static_cast<float>(Data[offset++]) / 255.0f;
        }
        
        // Get training flag from the input data
        bool training = true; // Default value
        if (offset < Size) {
            training = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Call update_bn_stats using torch::update_bn_stats
        torch::update_bn_stats(input_tensor, running_mean, running_var, momentum, training);
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
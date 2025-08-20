#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to initialize
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which initialization function to use
        if (offset >= Size) {
            return 0;
        }
        uint8_t init_type = Data[offset++];
        
        // Get a float value for parameters like gain, std, etc.
        float param_value = 0.01f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure param_value is reasonable (not NaN or Inf)
            if (std::isnan(param_value) || std::isinf(param_value)) {
                param_value = 0.01f;
            }
        }
        
        // Apply different initialization methods based on init_type
        switch (init_type % 10) {
            case 0:
                // Xavier uniform initialization
                torch::nn::init::xavier_uniform_(tensor, param_value);
                break;
            case 1:
                // Xavier normal initialization
                torch::nn::init::xavier_normal_(tensor, param_value);
                break;
            case 2:
                // Kaiming uniform initialization
                torch::nn::init::kaiming_uniform_(tensor, param_value);
                break;
            case 3:
                // Kaiming normal initialization
                torch::nn::init::kaiming_normal_(tensor, param_value);
                break;
            case 4:
                // Uniform initialization
                if (offset + sizeof(float) <= Size) {
                    float upper_bound;
                    std::memcpy(&upper_bound, Data + offset, sizeof(float));
                    if (std::isnan(upper_bound) || std::isinf(upper_bound)) {
                        upper_bound = 1.0f;
                    }
                    torch::nn::init::uniform_(tensor, param_value, upper_bound);
                } else {
                    torch::nn::init::uniform_(tensor);
                }
                break;
            case 5:
                // Normal initialization
                torch::nn::init::normal_(tensor, param_value, std::abs(param_value) + 0.1f);
                break;
            case 6:
                // Constant initialization
                torch::nn::init::constant_(tensor, param_value);
                break;
            case 7:
                // Ones initialization
                torch::nn::init::ones_(tensor);
                break;
            case 8:
                // Zeros initialization
                torch::nn::init::zeros_(tensor);
                break;
            case 9:
                // Eye initialization (only works for 2D tensors)
                if (tensor.dim() == 2) {
                    torch::nn::init::eye_(tensor);
                } else {
                    torch::nn::init::dirac_(tensor);
                }
                break;
        }
        
        // Try to access tensor values to ensure initialization worked
        if (tensor.numel() > 0) {
            auto accessor = tensor.accessor<float, 1>();
            volatile float value = accessor[0]; // Just to ensure tensor is accessed
            (void)value; // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
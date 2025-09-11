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
        
        // Get a value for parameters like gain, std, etc.
        float param_value = 0.01f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure param_value is not NaN or Inf
            if (std::isnan(param_value) || std::isinf(param_value)) {
                param_value = 0.01f;
            }
        }
        
        // Get a second parameter value if needed
        float param_value2 = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param_value2, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure param_value2 is not NaN or Inf
            if (std::isnan(param_value2) || std::isinf(param_value2)) {
                param_value2 = 0.0f;
            }
        }
        
        // Apply different initialization methods based on init_type
        switch (init_type % 12) {
            case 0:
                torch::nn::init::uniform_(tensor, -param_value, param_value);
                break;
            case 1:
                torch::nn::init::normal_(tensor, param_value, std::abs(param_value2) + 0.001f);
                break;
            case 2:
                torch::nn::init::constant_(tensor, param_value);
                break;
            case 3:
                torch::nn::init::ones_(tensor);
                break;
            case 4:
                torch::nn::init::zeros_(tensor);
                break;
            case 5:
                torch::nn::init::eye_(tensor);
                break;
            case 6:
                torch::nn::init::dirac_(tensor);
                break;
            case 7:
                torch::nn::init::xavier_uniform_(tensor, std::abs(param_value) + 0.001f);
                break;
            case 8:
                torch::nn::init::xavier_normal_(tensor, std::abs(param_value) + 0.001f);
                break;
            case 9:
                torch::nn::init::kaiming_uniform_(tensor, std::abs(param_value) + 0.001f, 
                                                 torch::kFanIn, torch::kLeakyReLU);
                break;
            case 10:
                torch::nn::init::kaiming_normal_(tensor, std::abs(param_value) + 0.001f, 
                                               torch::kFanOut, torch::kReLU);
                break;
            case 11:
                torch::nn::init::orthogonal_(tensor, std::abs(param_value) + 0.001f);
                break;
        }
        
        // Perform a simple operation to ensure the tensor is used
        auto sum = tensor.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

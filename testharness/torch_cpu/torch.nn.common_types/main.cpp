#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various common_types functionality
        
        // Test FanMode
        uint8_t fan_mode_selector = (offset < Size) ? Data[offset++] : 0;
        torch::kFanIn;
        torch::kFanOut;
        
        // Test NonlineairtyType - using string representations
        uint8_t nonlinearity_selector = (offset < Size) ? Data[offset++] : 0;
        std::string nonlinearity;
        switch (nonlinearity_selector % 5) {
            case 0:
                nonlinearity = "linear";
                break;
            case 1:
                nonlinearity = "conv1d";
                break;
            case 2:
                nonlinearity = "conv2d";
                break;
            case 3:
                nonlinearity = "conv3d";
                break;
            default:
                nonlinearity = "sigmoid";
                break;
        }
        
        // Test padding modes
        uint8_t padding_mode_selector = (offset < Size) ? Data[offset++] : 0;
        std::string padding_mode;
        switch (padding_mode_selector % 3) {
            case 0:
                padding_mode = "zeros";
                break;
            case 1:
                padding_mode = "reflect";
                break;
            default:
                padding_mode = "replicate";
                break;
        }
        
        // Test various combinations of these types
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            // Test with kaiming_uniform
            torch::nn::init::kaiming_uniform_(tensor, 0.0, torch::kFanIn, nonlinearity);
            
            // Test with xavier_normal
            torch::nn::init::xavier_normal_(tensor);
            
            // Test with orthogonal
            torch::nn::init::orthogonal_(tensor);
        }
        
        // Test enum conversions and other operations
        auto fan_mode_int = static_cast<int>(fan_mode_selector);
        auto nonlinearity_int = static_cast<int>(nonlinearity_selector);
        auto padding_mode_int = static_cast<int>(padding_mode_selector);
        
        // Create a tensor with these values to test operations
        if (offset + 3 <= Size) {
            std::vector<int64_t> shape = {3};
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            auto enum_tensor = torch::empty(shape, options);
            enum_tensor[0] = fan_mode_int;
            enum_tensor[1] = nonlinearity_int;
            enum_tensor[2] = padding_mode_int;
            
            // Perform some operations on the enum tensor
            auto result = enum_tensor + 1;
            auto result2 = result * 2;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
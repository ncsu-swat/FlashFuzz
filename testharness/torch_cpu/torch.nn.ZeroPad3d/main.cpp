#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 5D tensor (batch, channels, depth, height, width)
        // If not, reshape it to have at least 5 dimensions
        if (input_tensor.dim() < 5) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input_tensor.dim(); i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            
            // Add extra dimensions of size 1 to reach 5D
            while (new_shape.size() < 5) {
                new_shape.push_back(1);
            }
            
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Parse padding values from the remaining data
        std::vector<int64_t> padding(6, 0); // Default padding: [left, right, top, bottom, front, back]
        
        for (int i = 0; i < 6 && offset + sizeof(int32_t) <= Size; i++) {
            int32_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Allow negative padding to test edge cases
            padding[i] = pad_value;
        }
        
        // Create ZeroPad3d module
        auto pad_module = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(padding));
        
        // Apply padding
        torch::Tensor output_tensor = pad_module->forward(input_tensor);
        
        // Try to access elements of the output tensor to ensure computation is performed
        if (output_tensor.numel() > 0) {
            auto accessor = output_tensor.accessor<float, 5>();
            volatile float first_element = accessor[0][0][0][0][0];
            (void)first_element;
        }
        
        // Try alternative ways to create and use ZeroPad3d
        if (offset + sizeof(int32_t) <= Size) {
            int32_t padding_mode;
            std::memcpy(&padding_mode, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Test with single padding value
            int64_t single_pad = padding[0];
            auto single_pad_module = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(single_pad));
            torch::Tensor single_pad_output = single_pad_module->forward(input_tensor);
        }
        
        // Test with functional interface
        torch::Tensor functional_output = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                .mode(torch::kConstant)
                .value(0.0));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
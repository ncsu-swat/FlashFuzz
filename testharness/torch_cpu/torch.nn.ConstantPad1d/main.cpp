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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding values from the remaining data
        if (offset + 2 >= Size) {
            return 0;
        }
        
        // Extract padding configuration
        int64_t padding_left = static_cast<int64_t>(Data[offset++]);
        int64_t padding_right = static_cast<int64_t>(Data[offset++]);
        
        // Get value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create padding vector
        std::vector<int64_t> padding = {padding_left, padding_right};
        
        // Apply ConstantPad1d
        torch::Tensor output = torch::nn::functional::pad(
            input_tensor,
            padding,
            torch::nn::functional::PadFuncOptions().mode(torch::kConstant).value(pad_value)
        );
        
        // Force computation to ensure any potential errors are triggered
        output = output.contiguous();
        
        // Access elements to ensure computation is performed
        if (output.numel() > 0) {
            volatile float first_element = output.item<float>();
            (void)first_element;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

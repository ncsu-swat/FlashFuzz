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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left for padding values
        if (offset + 6 >= Size) {
            return 0;
        }
        
        // Extract padding values from the input data
        int64_t pad_left = static_cast<int64_t>(Data[offset++]) - 128;
        int64_t pad_right = static_cast<int64_t>(Data[offset++]) - 128;
        int64_t pad_top = static_cast<int64_t>(Data[offset++]) - 128;
        int64_t pad_bottom = static_cast<int64_t>(Data[offset++]) - 128;
        int64_t pad_front = static_cast<int64_t>(Data[offset++]) - 128;
        int64_t pad_back = static_cast<int64_t>(Data[offset++]) - 128;
        
        // Get value to pad with
        double pad_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create the ConstantPad3d module
        torch::nn::ConstantPad3d pad_module(
            torch::nn::ConstantPad3dOptions(
                {pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}, 
                pad_value
            )
        );
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input_tensor);
        
        // Perform some operations on the output to ensure it's used
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            if (sum.isnan().item<bool>()) {
                return 0;
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

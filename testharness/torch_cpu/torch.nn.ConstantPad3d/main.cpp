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
        std::vector<int64_t> padding(6);
        for (int i = 0; i < 6; i++) {
            if (offset < Size) {
                // Use the raw byte value to generate padding (can be negative)
                int8_t pad_value = static_cast<int8_t>(Data[offset++]);
                padding[i] = static_cast<int64_t>(pad_value);
            } else {
                padding[i] = 0;
            }
        }
        
        // Get value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Apply ConstantPad3d
        torch::nn::ConstantPad3d pad_module(padding, pad_value);
        torch::Tensor output = pad_module->forward(input_tensor);
        
        // Perform some operations on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum().item<float>();
            volatile float unused = sum; // Prevent optimization
            (void)unused;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

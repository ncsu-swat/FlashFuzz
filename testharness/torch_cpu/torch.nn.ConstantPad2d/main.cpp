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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 bytes left for padding values
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract padding values from the input data
        int64_t left = static_cast<int64_t>(Data[offset++]);
        int64_t right = static_cast<int64_t>(Data[offset++]);
        int64_t top = static_cast<int64_t>(Data[offset++]);
        int64_t bottom = static_cast<int64_t>(Data[offset++]);
        
        // Get value to pad with
        double pad_value = 0.0;
        if (Size - offset >= sizeof(float)) {
            float temp_value;
            std::memcpy(&temp_value, Data + offset, sizeof(float));
            pad_value = static_cast<double>(temp_value);
            offset += sizeof(float);
        }
        
        // Create the ConstantPad2d module
        torch::nn::ConstantPad2d pad(
            torch::nn::ConstantPad2dOptions(torch::ExpandingArray<4>({left, right, top, bottom}), pad_value)
        );
        
        // Apply padding
        torch::Tensor output = pad(input);
        
        // Optionally, perform some operations on the output to ensure it's used
        if (!output.sizes().empty()) {
            auto sum = output.sum();
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

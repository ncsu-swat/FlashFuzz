#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left for padding parameters
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Extract padding values from the input data
        int64_t padding_left = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_right = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_top = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_bottom = static_cast<int64_t>(Data[offset++]) % 10;
        
        // Extract value to pad with
        double pad_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create the ConstantPad2d module
        torch::nn::ConstantPad2d pad(
            torch::nn::ConstantPad2dOptions(
                {padding_left, padding_right, padding_top, padding_bottom}, 
                pad_value
            )
        );
        
        // Apply padding to the input tensor
        torch::Tensor output = pad->forward(input);
        
        // Optionally, perform some operations on the output to ensure it's used
        auto sum = output.sum().item<double>();
        if (std::isnan(sum) || std::isinf(sum)) {
            // Just to use the result and avoid compiler optimizing it away
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
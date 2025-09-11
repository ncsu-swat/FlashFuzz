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
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding parameters from the remaining data
        if (offset + 3 >= Size) {
            return 0;
        }
        
        // Extract padding values
        int64_t pad_left = static_cast<int64_t>(Data[offset++]);
        int64_t pad_right = static_cast<int64_t>(Data[offset++]);
        
        // Extract value to pad with
        float pad_value = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pad_value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Create the ConstantPad1d module
        torch::nn::ConstantPad1d pad(torch::nn::ConstantPad1dOptions({pad_left, pad_right}, pad_value));
        
        // Apply padding
        torch::Tensor output = pad(input);
        
        // Try different padding configurations
        if (offset + 2 < Size) {
            int64_t alt_pad = static_cast<int64_t>(Data[offset++]);
            torch::nn::ConstantPad1d single_pad(torch::nn::ConstantPad1dOptions(alt_pad, pad_value));
            torch::Tensor alt_output = single_pad(input);
        }
        
        // Try with different data types
        if (input.dtype() == torch::kFloat) {
            // Try with a different tensor type if possible
            try {
                torch::Tensor int_input = input.to(torch::kInt);
                torch::Tensor int_output = pad(int_input);
            } catch (...) {
                // Ignore conversion errors
            }
        }
        
        // Try with different dimensions if possible
        if (input.dim() > 1) {
            try {
                // Select first slice to get a lower dimensional tensor
                torch::Tensor slice = input.select(0, 0);
                torch::Tensor slice_output = pad(slice);
            } catch (...) {
                // Ignore dimension errors
            }
        }
        
        // Try with negative padding values
        if (offset + 2 < Size) {
            int64_t neg_pad_left = -static_cast<int64_t>(Data[offset++]);
            int64_t neg_pad_right = -static_cast<int64_t>(Data[offset++]);
            
            try {
                torch::nn::ConstantPad1d neg_pad(torch::nn::ConstantPad1dOptions({neg_pad_left, neg_pad_right}, pad_value));
                torch::Tensor neg_output = neg_pad(input);
            } catch (...) {
                // Ignore errors from negative padding
            }
        }
        
        // Try with very large padding values
        if (offset + 1 < Size) {
            int64_t large_pad = static_cast<int64_t>(Data[offset++]) * 1000;
            
            try {
                torch::nn::ConstantPad1d large_pad_module(torch::nn::ConstantPad1dOptions(large_pad, pad_value));
                torch::Tensor large_output = large_pad_module(input);
            } catch (...) {
                // Ignore errors from large padding
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

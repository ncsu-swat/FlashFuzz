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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // If not, reshape it to 5D
        if (input.dim() != 5) {
            // Get remaining bytes for reshaping
            if (offset + 5 > Size) {
                return 0;
            }
            
            // Extract dimensions for reshaping
            int64_t batch = (Data[offset] % 4) + 1;
            int64_t channels = (Data[offset + 1] % 4) + 1;
            int64_t depth = (Data[offset + 2] % 8) + 1;
            int64_t height = (Data[offset + 3] % 8) + 1;
            int64_t width = (Data[offset + 4] % 8) + 1;
            offset += 5;
            
            // Reshape tensor to 5D
            input = input.reshape({batch, channels, depth, height, width});
        }
        
        // Get padding parameters from the remaining data
        if (offset + 6 > Size) {
            return 0;
        }
        
        // Extract padding values (can be negative to test edge cases)
        int64_t pad_front = static_cast<int8_t>(Data[offset]);
        int64_t pad_back = static_cast<int8_t>(Data[offset + 1]);
        int64_t pad_top = static_cast<int8_t>(Data[offset + 2]);
        int64_t pad_bottom = static_cast<int8_t>(Data[offset + 3]);
        int64_t pad_left = static_cast<int8_t>(Data[offset + 4]);
        int64_t pad_right = static_cast<int8_t>(Data[offset + 5]);
        offset += 6;
        
        // Create ReflectionPad3d module
        torch::nn::ReflectionPad3d reflection_pad = nullptr;
        
        // Set padding
        if (offset < Size && Data[offset] % 2 == 0) {
            // Use single value padding
            int64_t padding = static_cast<int8_t>(Data[offset]);
            reflection_pad = torch::nn::ReflectionPad3d(padding);
        } else {
            // Use tuple padding (front, back, top, bottom, left, right)
            reflection_pad = torch::nn::ReflectionPad3d(
                torch::nn::ReflectionPad3dOptions(
                    {pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right}
                )
            );
        }
        
        // Apply padding
        torch::Tensor output = reflection_pad->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum().item<float>();
        
        // Try different padding configurations
        if (offset + 1 < Size) {
            // Try another padding configuration
            int64_t alt_padding = static_cast<int8_t>(Data[offset]);
            torch::nn::ReflectionPad3d alt_pad(alt_padding);
            torch::Tensor alt_output = alt_pad->forward(input);
            auto alt_sum = alt_output.sum().item<float>();
            
            // Combine results to prevent optimization removing the computation
            if (std::isnan(sum + alt_sum)) {
                throw std::runtime_error("NaN detected in output");
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

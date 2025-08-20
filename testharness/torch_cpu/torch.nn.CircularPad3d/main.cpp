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
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // If not, reshape it to 5D
        if (input_tensor.dim() < 3) {
            std::vector<int64_t> new_shape = {1, 1, 1, 1, 1};
            for (int i = 0; i < input_tensor.dim(); i++) {
                new_shape[5 - input_tensor.dim() + i] = input_tensor.size(i);
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        else if (input_tensor.dim() > 5) {
            // Flatten extra dimensions
            std::vector<int64_t> new_shape = {1, 1, 1, 1, 1};
            int64_t product = 1;
            for (int i = 0; i < input_tensor.dim() - 5; i++) {
                product *= input_tensor.size(i);
            }
            new_shape[0] = product;
            for (int i = 0; i < std::min(4, static_cast<int>(input_tensor.dim())); i++) {
                new_shape[i+1] = input_tensor.size(input_tensor.dim() - 4 + i);
            }
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Parse padding values from the remaining data
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative padding values to test edge cases
            padding[i] = pad_value;
        }
        
        // Apply circular padding using torch::nn::functional::pad
        torch::Tensor output;
        
        // Try different padding configurations
        if (offset + 1 <= Size) {
            uint8_t pad_config = Data[offset++];
            
            if (pad_config % 3 == 0) {
                // Single integer padding
                int64_t pad_value = padding[0];
                output = torch::nn::functional::pad(input_tensor, 
                    torch::nn::functional::PadFuncOptions({pad_value, pad_value, pad_value, pad_value, pad_value, pad_value})
                    .mode(torch::kCircular));
            }
            else if (pad_config % 3 == 1) {
                // Tuple of 6 integers (left, right, top, bottom, front, back)
                output = torch::nn::functional::pad(input_tensor, 
                    torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]})
                    .mode(torch::kCircular));
            }
            else {
                // Tuple of 3 pairs (depth_padding, height_padding, width_padding)
                output = torch::nn::functional::pad(input_tensor, 
                    torch::nn::functional::PadFuncOptions({padding[4], padding[5], padding[2], padding[3], padding[0], padding[1]})
                    .mode(torch::kCircular));
            }
        }
        else {
            // Default padding if not enough data
            output = torch::nn::functional::pad(input_tensor, 
                torch::nn::functional::PadFuncOptions({1, 1, 1, 1, 1, 1})
                .mode(torch::kCircular));
        }
        
        // Force computation to catch any errors
        output.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
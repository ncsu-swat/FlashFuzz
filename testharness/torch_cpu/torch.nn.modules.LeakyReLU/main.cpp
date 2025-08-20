#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for negative_slope parameter
        if (Size < 1) {
            return 0;
        }
        
        // Parse negative_slope parameter from the first byte
        float negative_slope = static_cast<float>(Data[0]) / 255.0f;
        offset++;
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a default tensor
            input = torch::randn({2, 3});
        }
        
        // Create LeakyReLU module with the parsed negative_slope
        torch::nn::LeakyReLU leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(negative_slope));
        
        // Apply LeakyReLU to the input tensor
        torch::Tensor output = leaky_relu->forward(input);
        
        // Try inplace version as well if we have enough data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_clone = input.clone();
            torch::nn::LeakyReLU leaky_relu_inplace(torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(true));
            torch::Tensor output_inplace = leaky_relu_inplace->forward(input_clone);
        }
        
        // Try with different data types if we have more data
        if (offset + 1 < Size) {
            // Create a tensor with a different data type
            size_t new_offset = offset;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // Apply LeakyReLU to the second input tensor
            torch::Tensor output2 = leaky_relu->forward(input2);
        }
        
        // Try with extreme negative_slope values if we have more data
        if (offset + 2 < Size) {
            float extreme_slope;
            if (Data[offset] % 3 == 0) {
                // Very small negative slope
                extreme_slope = 1e-10f;
            } else if (Data[offset] % 3 == 1) {
                // Very large negative slope
                extreme_slope = 1e10f;
            } else {
                // Negative negative slope (should still work)
                extreme_slope = -static_cast<float>(Data[offset]) / 255.0f;
            }
            
            torch::nn::LeakyReLU extreme_leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(extreme_slope));
            torch::Tensor extreme_output = extreme_leaky_relu->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
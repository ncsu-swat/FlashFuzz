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
        
        // Alternative way to apply LeakyReLU using functional API
        torch::Tensor output_functional = torch::leaky_relu(input, negative_slope);
        
        // Try inplace version if we have enough data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_clone = input.clone();
            torch::leaky_relu_(input_clone, negative_slope);
        }
        
        // Try with different tensor types if we have more data
        if (offset + 1 < Size) {
            torch::ScalarType dtype = fuzzer_utils::parseDataType(Data[offset++]);
            
            // Only try with floating point types as LeakyReLU is typically used with them
            if (dtype == torch::kFloat || dtype == torch::kDouble || 
                dtype == torch::kHalf || dtype == torch::kBFloat16) {
                torch::Tensor input_cast = input.to(dtype);
                torch::Tensor output_cast = leaky_relu->forward(input_cast);
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
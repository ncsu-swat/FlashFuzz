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
        
        // Extract parameters for LinearReLU
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar tensors, use a small value
            in_features = 4;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            // Default value if not enough data
            out_features = 4;
        }
        
        // Create a regular Linear layer followed by ReLU as approximation
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
        
        // Apply the module to the input tensor
        torch::Tensor output;
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // For scalar input, reshape to 1D
            torch::Tensor linear_out = linear(input.reshape({1}));
            output = torch::relu(linear_out);
        } else if (input.dim() == 1) {
            // For 1D input, make sure it has the right size
            if (input.size(0) != in_features) {
                input = input.reshape({1, -1});
                if (input.size(1) != in_features) {
                    // Resize if needed
                    input = torch::zeros({1, in_features}, input.options());
                }
            }
            torch::Tensor linear_out = linear(input);
            output = torch::relu(linear_out);
        } else {
            // For N-D input, make sure the last dimension matches in_features
            if (input.size(-1) != in_features) {
                // Resize the last dimension if needed
                std::vector<int64_t> new_shape = input.sizes().vec();
                new_shape.back() = in_features;
                input = torch::zeros(new_shape, input.options());
            }
            torch::Tensor linear_out = linear(input);
            output = torch::relu(linear_out);
        }
        
        // Try with bias=false
        if (offset < Size) {
            torch::nn::Linear linear_no_bias(torch::nn::LinearOptions(in_features, out_features).bias(false));
            
            torch::Tensor linear_out_no_bias = linear_no_bias(input);
            torch::Tensor output_no_bias = torch::relu(linear_out_no_bias);
        }
        
        // Try with different dtype
        if (offset < Size) {
            // Create a float input tensor
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor float_linear_out = linear(float_input);
            torch::Tensor float_output = torch::relu(float_linear_out);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

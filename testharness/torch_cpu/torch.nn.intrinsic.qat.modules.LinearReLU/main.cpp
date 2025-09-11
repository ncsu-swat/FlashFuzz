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
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        
        // Get in_features from the last dimension of input if it exists
        if (input.dim() > 0) {
            in_features = input.size(-1);
        } else {
            // For scalar input, use a small value
            in_features = 1;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features reasonable but allow edge cases
            out_features = std::abs(out_features) % 128 + 1;
        } else {
            // Default value if not enough data
            out_features = 4;
        }
        
        // Create Linear module followed by ReLU
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
        torch::nn::ReLU relu;
        
        // Reshape input if needed to match Linear requirements
        if (input.dim() == 0) {
            // Convert scalar to 1D tensor
            input = input.reshape({1});
        }
        
        // For higher dimensional tensors, ensure the last dimension matches in_features
        if (input.dim() >= 2) {
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[input.dim() - 1] = in_features;
            input = input.reshape(new_shape);
        } else if (input.dim() == 1 && input.size(0) != in_features) {
            // For 1D tensor, reshape to match in_features
            input = input.reshape({in_features});
        }
        
        // Apply the Linear operation followed by ReLU
        torch::Tensor linear_output = linear(input);
        torch::Tensor output = relu(linear_output);
        
        // Try to access some properties to ensure the operation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try to get the weight and bias
        auto weight = linear->weight;
        auto bias = linear->bias;
        
        // Try different training modes
        if (offset + 1 < Size) {
            uint8_t train_flag = Data[offset++];
            if (train_flag % 2 == 0) {
                linear->train();
                relu->train();
            } else {
                linear->eval();
                relu->eval();
            }
        }
        
        // Try to access weight gradients if in training mode
        if (offset < Size && Data[offset] % 2 == 0) {
            if (weight.requires_grad()) {
                auto grad = weight.grad();
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

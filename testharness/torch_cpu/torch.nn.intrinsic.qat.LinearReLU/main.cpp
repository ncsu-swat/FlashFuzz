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
        
        // Early exit if not enough data
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar input, use a small value
            in_features = 4;
        }
        
        // Get out_features from remaining data if available
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4;
        }
        
        // Get bias parameter if data available
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create a regular Linear module followed by ReLU
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Reshape input if needed to match expected input shape for Linear
        if (input.dim() == 0) {
            // Convert scalar to 1D tensor
            input = input.reshape({1, in_features});
        } else if (input.dim() == 1) {
            // Convert 1D tensor to 2D tensor
            input = input.reshape({1, input.size(0)});
            
            // If the last dimension doesn't match in_features, reshape it
            if (input.size(-1) != in_features) {
                input = input.reshape({1, in_features});
            }
        } else {
            // For higher dimensions, ensure the last dimension matches in_features
            auto sizes = input.sizes().vec();
            if (sizes.back() != in_features) {
                sizes.back() = in_features;
                input = input.reshape(sizes);
            }
        }
        
        // Apply the Linear operation followed by ReLU
        torch::Tensor linear_output = linear(input);
        torch::Tensor output = torch::relu(linear_output);
        
        // Try to access some properties to ensure computation happened
        auto weight = linear->weight;
        auto output_size = output.sizes();
        
        // Additional operations to test the tensor
        if (output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

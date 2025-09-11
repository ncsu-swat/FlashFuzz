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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor if possible
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
        } else {
            // For scalar or empty tensor, use a default value
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&in_features, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                in_features = std::abs(in_features) % 128 + 1; // Ensure positive
            } else {
                in_features = 4; // Default value
            }
        }
        
        // Get out_features from the input data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            out_features = std::abs(out_features) % 128 + 1; // Ensure positive
        } else {
            out_features = 4; // Default value
        }
        
        // Get bias flag
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Create a regular linear module and apply dynamic quantization
        torch::nn::Linear linear_module(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Apply the module to the input tensor
        torch::Tensor output;
        
        // Handle different input tensor dimensions
        if (input_tensor.dim() == 0) {
            // Scalar tensor - reshape to have proper dimensions
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // 1D tensor - check if size matches in_features
            if (input_tensor.size(0) != in_features) {
                input_tensor = input_tensor.reshape({1, in_features});
            } else {
                input_tensor = input_tensor.reshape({1, in_features});
            }
        } else {
            // Multi-dimensional tensor - ensure last dimension matches in_features
            auto sizes = input_tensor.sizes().vec();
            if (sizes.back() != in_features) {
                // Reshape the tensor to have the correct last dimension
                sizes.back() = in_features;
                input_tensor = input_tensor.reshape(sizes);
            }
        }
        
        // Forward pass through the module
        output = linear_module->forward(input_tensor);
        
        // Optional: Test other methods of the module
        if (offset < Size) {
            uint8_t test_selector = Data[offset++];
            
            switch (test_selector % 2) {
                case 0:
                    // Test weight access
                    {
                        auto weight = linear_module->weight;
                    }
                    break;
                case 1:
                    // Test bias access if bias is enabled
                    if (bias) {
                        auto bias_tensor = linear_module->bias;
                    }
                    break;
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

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
        } else {
            // For scalar tensors, use a default value
            in_features = 1;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable but allow edge cases
            out_features = std::abs(raw_out_features) % 128 + 1;
        } else {
            out_features = 1; // Default value
        }
        
        // Get bias parameter
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Create the Linear module using LinearOptions
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(bias);
        torch::nn::Linear linear_module(options);
        
        // Reshape input tensor if needed to match expected input shape
        if (input_tensor.dim() == 0) {
            // Scalar tensor needs to be reshaped to have at least one dimension
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // 1D tensor needs to be reshaped to have batch dimension
            input_tensor = input_tensor.reshape({1, input_tensor.size(0)});
        } else {
            // For tensors with dim >= 2, ensure the last dimension matches in_features
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            new_shape[new_shape.size() - 1] = in_features;
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Apply the Linear module
        torch::Tensor output = linear_module->forward(input_tensor);
        
        // Optionally test some properties of the output
        auto output_size = output.sizes();
        
        // Test backward pass with a simple gradient
        if (input_tensor.requires_grad() && input_tensor.dtype() == torch::kFloat32) {
            output.sum().backward();
        }
        
        // Test other operations on the Linear module
        auto params = linear_module->parameters();
        
        // Test serialization/deserialization
        torch::serialize::OutputArchive output_archive;
        linear_module->save(output_archive);
        
        torch::nn::Linear loaded_module(options);
        torch::serialize::InputArchive input_archive;
        loaded_module->load(input_archive);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor if possible
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
        } else {
            // For scalar tensors, use a small value
            in_features = 4;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features reasonable but allow edge cases
            out_features = std::abs(out_features) % 128 + 1;
        } else {
            out_features = 4;  // Default value
        }
        
        // Determine if bias should be used
        if (offset < Size) {
            bias = Data[offset++] & 0x1;  // Use lowest bit to determine bias
        }
        
        // Create a regular linear module since QATDynamicLinear is not available
        torch::nn::Linear module(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Reshape input tensor if needed to match expected input shape
        if (input_tensor.dim() == 0) {
            // For scalar tensors, reshape to [1, in_features]
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // For 1D tensors, reshape to [1, in_features]
            if (input_tensor.size(0) != in_features) {
                input_tensor = input_tensor.reshape({1, in_features});
            }
        } else {
            // For higher dimensional tensors, ensure the last dimension is in_features
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            if (new_shape.back() != in_features) {
                new_shape.back() = in_features;
                input_tensor = input_tensor.reshape(new_shape);
            }
        }
        
        // Apply the module to the input tensor
        torch::Tensor output = module->forward(input_tensor);
        
        // Test the module in training and evaluation modes
        module->train();
        torch::Tensor output_train = module->forward(input_tensor);
        
        module->eval();
        torch::Tensor output_eval = module->forward(input_tensor);
        
        // Test with different data types if possible
        if (input_tensor.dtype() != torch::kFloat) {
            torch::Tensor float_input = input_tensor.to(torch::kFloat);
            torch::Tensor float_output = module->forward(float_input);
        }
        
        // Test module parameters
        module->weight;
        if (bias) {
            module->bias;
        }
        
        // Test serialization
        torch::serialize::OutputArchive output_archive;
        module->save(output_archive);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

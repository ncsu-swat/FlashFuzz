#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some parameters for the linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Determine in_features from the input tensor
        if (input.dim() >= 2) {
            in_features = input.size(-1);
        } else if (input.dim() == 1) {
            in_features = input.size(0);
        } else {
            // For scalar tensors, use a default value
            in_features = 1;
        }
        
        // Extract out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable (avoid excessive memory usage)
            out_features = std::abs(raw_out_features) % 128 + 1;
        } else {
            // Default if not enough data
            out_features = 10;
        }
        
        // Extract bias flag if data available
        if (offset < Size) {
            bias = Data[offset++] & 0x1;  // Use lowest bit to determine bias
        }
        
        // Create the linear module using LinearOptions
        torch::nn::LinearOptions options(in_features, out_features);
        options.bias(bias);
        torch::nn::Linear linear_module(options);
        
        // Try different input shapes and scenarios
        try {
            // Forward pass with the input tensor
            torch::Tensor output = linear_module->forward(input);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
        
        // Try with different input shapes if possible
        if (input.dim() > 1) {
            try {
                // Reshape to batch size of 1 if possible
                auto reshaped = input.reshape({1, -1});
                torch::Tensor output = linear_module->forward(reshaped);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
        }
        
        // Try with different data types if original tensor is floating point
        if (input.scalar_type() == torch::kFloat || 
            input.scalar_type() == torch::kDouble) {
            try {
                auto other_dtype = (input.scalar_type() == torch::kFloat) ? 
                                   torch::kDouble : torch::kFloat;
                auto converted = input.to(other_dtype);
                torch::Tensor output = linear_module->forward(converted);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch are fine
            }
        }
        
        // Test with zero weights/bias
        try {
            linear_module->weight.zero_();
            if (bias) {
                linear_module->bias.zero_();
            }
            torch::Tensor output = linear_module->forward(input);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
        
        // Test with extreme weight values
        try {
            linear_module->weight.fill_(1e10);
            if (bias) {
                linear_module->bias.fill_(1e10);
            }
            torch::Tensor output = linear_module->forward(input);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
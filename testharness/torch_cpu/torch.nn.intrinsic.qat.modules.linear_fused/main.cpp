#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for the linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Determine in_features from input tensor
        if (input.dim() >= 2) {
            in_features = input.size(-1);
        } else if (input.dim() == 1) {
            in_features = input.size(0);
        } else {
            in_features = 1;
        }
        
        // Determine out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 100 + 1;
        } else {
            out_features = 10; // Default value
        }
        
        // Determine if bias should be used
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create a regular Linear layer followed by ReLU
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        torch::nn::ReLU relu;
        
        // Set to training mode
        linear->train();
        
        // Forward pass
        torch::Tensor output;
        
        // Reshape input if needed
        if (input.dim() == 0) {
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        }
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat) {
            input = input.to(torch::kFloat);
        }
        
        // Forward pass through linear then relu
        torch::Tensor linear_output = linear->forward(input);
        output = relu->forward(linear_output);
        
        // Test with different batch sizes
        if (offset + 1 < Size && input.dim() >= 2) {
            int64_t new_batch_size = Data[offset++] % 5 + 1;
            
            try {
                torch::Tensor resized_input = input.expand({new_batch_size, -1});
                torch::Tensor linear_out = linear->forward(resized_input);
                torch::Tensor new_output = relu->forward(linear_out);
            } catch (const std::exception& e) {
                // Expansion might fail for some inputs
            }
        }
        
        // Test backward pass
        if (output.requires_grad()) {
            try {
                auto grad_output = torch::ones_like(output);
                output.backward(grad_output);
            } catch (const std::exception& e) {
                // Backward might fail for some inputs
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
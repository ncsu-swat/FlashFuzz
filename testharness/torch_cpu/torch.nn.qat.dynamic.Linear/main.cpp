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
        
        // Extract parameters for Linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar input, use a small value
            in_features = 1;
        }
        
        // Get out_features from remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4; // Default value
        }
        
        // Get bias flag from remaining data
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create regular linear layer (QAT dynamic linear is not available in C++ frontend)
        torch::nn::Linear linear(
            torch::nn::LinearOptions(in_features, out_features).bias(bias)
        );
        
        // Apply the linear layer to the input
        torch::Tensor output;
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // Scalar input - reshape to 1D
            output = linear(input.reshape({1}));
        } else if (input.dim() == 1) {
            // 1D input
            output = linear(input);
        } else {
            // Multi-dimensional input
            output = linear(input);
        }
        
        // Try to get scale and zero_point for quantization testing
        if (offset + 2 * sizeof(float) <= Size) {
            float scale, zero_point;
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&zero_point, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure scale is positive
            scale = std::abs(scale) + 1e-5;
            
            // Try to quantize the input
            try {
                auto quantized_input = torch::quantize_per_tensor(
                    input.to(torch::kFloat), scale, static_cast<int>(zero_point), torch::kQUInt8);
                
                // Dequantize and run through linear layer
                auto dequantized_input = torch::dequantize(quantized_input);
                auto quantized_output = linear(dequantized_input);
            } catch (...) {
                // Ignore quantization errors
            }
        }
        
        // Test weight and bias access
        auto weight = linear->weight();
        if (bias) {
            auto bias_tensor = linear->bias();
        }
        
        // Test state dict
        auto state_dict = linear->state_dict();
        
        // Test other methods
        linear->reset_parameters();
        linear->forward(input);
        
        // Test different modes
        linear->train();
        linear->eval();
        
        // Test cloning
        auto cloned = linear->clone();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - must be 5D for BatchNorm3d
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor is 5D (required for BatchNorm3d)
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                // Add dimensions if needed
                new_shape = input.sizes().vec();
                while (new_shape.size() < 5) {
                    new_shape.push_back(1);
                }
            } else if (input.dim() > 5) {
                // Collapse extra dimensions
                new_shape.push_back(input.size(0)); // N
                new_shape.push_back(input.size(1)); // C
                new_shape.push_back(1);             // D
                new_shape.push_back(1);             // H
                new_shape.push_back(1);             // W
            }
            input = input.reshape(new_shape);
        }
        
        // Get the number of channels (second dimension)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            num_features = 1; // Ensure at least one channel
        }
        
        // Parse additional parameters from the input data
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 8 <= Size) {
            // Extract eps (ensure it's positive)
            double raw_eps;
            std::memcpy(&raw_eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            eps = std::abs(raw_eps);
            if (eps == 0.0) eps = 1e-5;
            
            // Extract momentum (ensure it's between 0 and 1)
            double raw_momentum;
            std::memcpy(&raw_momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            momentum = std::abs(raw_momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Create scale and zero_point for quantization
        auto scale = torch::ones({num_features});
        auto zero_point = torch::zeros({num_features}, torch::kInt);
        
        // Create running_mean and running_var
        auto running_mean = torch::zeros({num_features});
        auto running_var = torch::ones({num_features});
        
        // Create weight and bias
        auto weight = torch::ones({num_features});
        auto bias = torch::zeros({num_features});
        
        // Quantize the input tensor
        auto q_input = torch::quantize_per_channel(
            input,
            scale,
            zero_point,
            1,  // axis (channel dimension)
            torch::kQUInt8
        );
        
        // Apply the quantized batch_norm3d operation directly
        auto output = torch::quantized_batch_norm(
            q_input,
            weight,
            bias,
            running_mean,
            running_var,
            false,  // training
            momentum,
            eps,
            scale,
            zero_point
        );
        
        // Dequantize the output for verification
        auto dq_output = output.dequantize();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

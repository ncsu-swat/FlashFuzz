#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            // Expand dimensions to make it 4D
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Ensure input is float type for normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get number of channels from input tensor
        int64_t num_features = input.size(1);
        
        // Ensure num_features is valid (at least 1)
        if (num_features < 1) {
            return 0;
        }
        
        // Parse parameters from the input data
        bool affine = offset < Size ? (Data[offset++] & 0x1) : false;
        bool track_running_stats = offset < Size ? (Data[offset++] & 0x1) : false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (eps == 0.0 || std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
            if (eps > 1.0) eps = 1e-5; // Keep eps small
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Create InstanceNorm2d module with inferred num_features
        // This simulates lazy behavior by determining num_features from the input
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output;
        try {
            output = instance_norm->forward(input);
        } catch (const c10::Error&) {
            // Expected failures for invalid shapes/inputs
            return 0;
        }
        
        // Force computation to ensure lazy tensors are materialized
        output = output.clone();
        
        // Access some properties to ensure computation is done
        auto output_size = output.sizes();
        (void)output_size;
        
        // Verify output shape matches input shape
        if (output.sizes() != input.sizes()) {
            // Shape mismatch would be a bug
            std::cerr << "Shape mismatch!" << std::endl;
        }
        
        // Test with a second input to verify the module works with different spatial dims
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input2.dim() < 4) {
                while (input2.dim() < 4) {
                    input2 = input2.unsqueeze(0);
                }
            }
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Ensure same number of channels as first input
            if (input2.size(1) == num_features) {
                try {
                    torch::Tensor output2 = instance_norm->forward(input2);
                    output2 = output2.clone();
                } catch (const c10::Error&) {
                    // Expected for shape mismatches
                }
            }
        }
        
        // Test eval mode
        instance_norm->eval();
        try {
            torch::Tensor eval_output = instance_norm->forward(input);
            eval_output = eval_output.clone();
        } catch (const c10::Error&) {
            // May fail in eval mode with track_running_stats=false
        }
        
        // Test train mode again
        instance_norm->train();
        try {
            torch::Tensor train_output = instance_norm->forward(input);
            train_output = train_output.clone();
        } catch (const c10::Error&) {
            // Expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
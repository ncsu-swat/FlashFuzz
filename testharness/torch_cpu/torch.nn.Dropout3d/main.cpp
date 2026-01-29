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
        
        // Need at least a few bytes for tensor creation and dropout parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Dropout3d expects 5D input: (N, C, D, H, W)
        // Reshape tensor to be 5D regardless of original shape
        std::vector<int64_t> new_shape;
        int64_t numel = input.numel();
        if (numel == 0) {
            return 0; // Skip empty tensors
        }
        
        if (input.dim() == 0) {
            // Scalar tensor, reshape to [1, 1, 1, 1, 1]
            new_shape = {1, 1, 1, 1, 1};
        } else if (input.dim() == 1) {
            // 1D tensor, reshape to [1, 1, numel, 1, 1]
            new_shape = {1, 1, numel, 1, 1};
        } else if (input.dim() == 2) {
            // 2D tensor, reshape to [1, 1, input.size(0), input.size(1), 1]
            new_shape = {1, 1, input.size(0), input.size(1), 1};
        } else if (input.dim() == 3) {
            // 3D tensor, reshape to [1, input.size(0), input.size(1), input.size(2), 1]
            new_shape = {1, input.size(0), input.size(1), input.size(2), 1};
        } else if (input.dim() == 4) {
            // 4D tensor, reshape to [input.size(0), input.size(1), input.size(2), input.size(3), 1]
            new_shape = {input.size(0), input.size(1), input.size(2), input.size(3), 1};
        } else {
            // Already 5D or more, use first 5 dimensions or flatten extras
            new_shape = {input.size(0), input.size(1), input.size(2), input.size(3), input.size(4)};
        }
        
        input = input.reshape(new_shape);
        
        // Ensure input is floating point for dropout
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract dropout probability from input data
        double p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            float p_raw;
            std::memcpy(&p_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Convert to a valid probability value [0, 1)
            p = std::abs(p_raw);
            p = p - std::floor(p); // Get fractional part
            // Clamp to valid range
            if (p >= 1.0) p = 0.999;
            if (std::isnan(p) || std::isinf(p)) p = 0.5;
        }
        
        // Extract inplace flag from input data
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) != 0;
        }
        
        // Extract training mode from input data
        bool training = true;
        if (offset < Size) {
            training = (Data[offset++] & 0x01) != 0;
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout(torch::nn::Dropout3dOptions().p(p).inplace(inplace));
        
        // Set training mode
        dropout->train(training);
        
        // Clone input if inplace to avoid issues with subsequent operations
        torch::Tensor input_for_main = inplace ? input.clone() : input;
        
        // Apply dropout to input tensor
        torch::Tensor output = dropout(input_for_main);
        
        // Test edge case: zero probability (should keep all values)
        try {
            torch::nn::Dropout3d dropout_zero(torch::nn::Dropout3dOptions().p(0.0).inplace(false));
            dropout_zero->train(training);
            torch::Tensor output_zero = dropout_zero(input.clone());
        } catch (...) {
            // Silently ignore edge case failures
        }
        
        // Test edge case: high probability
        try {
            torch::nn::Dropout3d dropout_high(torch::nn::Dropout3dOptions().p(0.99).inplace(false));
            dropout_high->train(true);
            torch::Tensor output_high = dropout_high(input.clone());
        } catch (...) {
            // Silently ignore edge case failures
        }
        
        // Test in eval mode (should be identity function)
        try {
            torch::nn::Dropout3d dropout_eval(torch::nn::Dropout3dOptions().p(p).inplace(false));
            dropout_eval->eval();
            torch::Tensor output_eval = dropout_eval(input.clone());
        } catch (...) {
            // Silently ignore edge case failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
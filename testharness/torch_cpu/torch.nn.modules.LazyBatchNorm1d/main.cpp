#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm1d from the remaining data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and reasonable
            eps = std::abs(eps);
            if (eps == 0.0 || std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
            if (eps > 1.0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Reshape input if needed to match BatchNorm1d requirements
        // BatchNorm1d expects input of shape [N, C] or [N, C, L]
        if (input.dim() == 1) {
            // Add batch dimension: [C] -> [1, C]
            input = input.unsqueeze(0);
        } else if (input.dim() > 3) {
            // Flatten extra dimensions into the length dimension
            auto sizes = input.sizes().vec();
            int64_t new_length = 1;
            for (size_t i = 2; i < sizes.size(); ++i) {
                new_length *= sizes[i];
            }
            input = input.reshape({sizes[0], sizes[1], new_length});
        }
        
        // Ensure we have at least 1 feature (channel dimension)
        if (input.dim() < 2 || input.size(1) == 0) {
            return 0;
        }
        
        // Get number of features from input tensor (channel dimension)
        int64_t num_features = input.size(1);
        
        // Ensure input is float type for batch normalization
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create BatchNorm1d module with inferred num_features
        // Note: LazyBatchNorm1d is Python-only, so we use BatchNorm1d in C++
        auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features)
                                         .eps(eps)
                                         .momentum(momentum)
                                         .affine(affine)
                                         .track_running_stats(track_running_stats));
        
        // First forward pass
        torch::Tensor output;
        try {
            output = bn->forward(input);
        } catch (const c10::Error&) {
            // Shape mismatch or other tensor errors are expected
            return 0;
        }
        
        // Access parameters
        if (affine && bn->weight.defined()) {
            auto w = bn->weight.data();
            (void)w;
        }
        if (affine && bn->bias.defined()) {
            auto b = bn->bias.data();
            (void)b;
        }
        if (track_running_stats && bn->running_mean.defined()) {
            auto rm = bn->running_mean.data();
            (void)rm;
        }
        if (track_running_stats && bn->running_var.defined()) {
            auto rv = bn->running_var.data();
            (void)rv;
        }
        
        // Test the module in training mode
        bn->train();
        try {
            torch::Tensor train_output = bn->forward(input);
            (void)train_output;
        } catch (const c10::Error&) {
            // Expected errors in training mode
        }
        
        // Test the module in eval mode
        bn->eval();
        try {
            torch::Tensor eval_output = bn->forward(input);
            (void)eval_output;
        } catch (const c10::Error&) {
            // Expected errors in eval mode
        }
        
        // Test with a different input to exercise running stats update
        if (offset + 4 <= Size) {
            size_t offset2 = 0;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Reshape to match expected dimensions
            if (input2.dim() == 1) {
                input2 = input2.unsqueeze(0);
            } else if (input2.dim() > 3) {
                auto sizes = input2.sizes().vec();
                int64_t new_length = 1;
                for (size_t i = 2; i < sizes.size(); ++i) {
                    new_length *= sizes[i];
                }
                input2 = input2.reshape({sizes[0], sizes[1], new_length});
            }
            
            if (!input2.is_floating_point()) {
                input2 = input2.to(torch::kFloat32);
            }
            
            // Ensure input2 has the same number of features as input
            if (input2.dim() >= 2 && input2.size(1) == num_features) {
                // Try to forward with the second input
                bn->train();
                try {
                    torch::Tensor output2 = bn->forward(input2);
                    (void)output2;
                } catch (const c10::Error&) {
                    // Shape mismatch expected if dimensions differ
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
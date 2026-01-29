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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // group_norm requires at least 2D input (N, C, ...)
        if (input.dim() < 2) {
            return 0;
        }
        
        // Ensure we have a reasonable number of channels
        int64_t num_channels = input.size(1);
        if (num_channels <= 0 || num_channels > 1024) {
            return 0;
        }
        
        // Extract parameters for group_norm
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Parse number of groups
        uint8_t groups_byte = Data[offset++];
        // Ensure num_groups is between 1 and the number of channels
        int64_t num_groups = (groups_byte % num_channels) + 1;
        // Ensure num_channels is divisible by num_groups
        while (num_channels % num_groups != 0 && num_groups > 1) {
            num_groups--;
        }
        
        // Parse epsilon value from a byte (scaled to reasonable range)
        uint8_t eps_byte = Data[offset++];
        double epsilon = 1e-5 + (eps_byte / 255.0) * 1e-3; // Range: 1e-5 to ~1e-3
        
        // Decide whether to use weight/bias based on fuzzer data
        bool use_affine = false;
        if (offset < Size) {
            use_affine = (Data[offset++] % 2) == 1;
        }
        
        // Create weight and bias tensors
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (use_affine) {
            // Use float dtype for weight and bias to avoid issues
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Initialize weight to ones with some variation
            weight = torch::ones({num_channels}, options);
            if (offset + num_channels <= Size) {
                for (int64_t i = 0; i < num_channels && offset < Size; i++, offset++) {
                    // Scale byte to range [0.5, 1.5]
                    weight[i] = 0.5f + (Data[offset] / 255.0f);
                }
            }
            
            // Initialize bias to zeros with some variation
            bias = torch::zeros({num_channels}, options);
            if (offset + num_channels <= Size) {
                for (int64_t i = 0; i < num_channels && offset < Size; i++, offset++) {
                    // Scale byte to range [-0.5, 0.5]
                    bias[i] = (Data[offset] / 255.0f) - 0.5f;
                }
            }
        }
        
        // Convert input to float if needed (group_norm works best with float)
        torch::Tensor float_input = input.to(torch::kFloat32);
        
        // Apply group_norm operation
        torch::Tensor output;
        
        try {
            // Handle different cases based on available parameters
            if (use_affine && weight.defined() && bias.defined()) {
                output = torch::group_norm(float_input, num_groups, weight, bias, epsilon);
            } else {
                output = torch::group_norm(float_input, num_groups, {}, {}, epsilon);
            }
        } catch (const c10::Error &e) {
            // Silently catch expected errors (shape mismatches, etc.)
            return 0;
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Ensure the operation is not optimized away
        volatile float result = sum.item<float>();
        (void)result;
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}
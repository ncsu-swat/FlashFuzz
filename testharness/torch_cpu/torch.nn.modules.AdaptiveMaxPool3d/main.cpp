#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::tie

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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 4 dimensions (batch, channels, D, H, W)
        // AdaptiveMaxPool3d accepts 4D (C, D, H, W) or 5D (N, C, D, H, W) input
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            
            // Add missing dimensions to reach at least 4D
            while (new_shape.size() < 4) {
                new_shape.push_back(1);
            }
            
            input = input.reshape(new_shape);
        }
        
        // Ensure spatial dimensions are at least 1
        auto input_sizes = input.sizes().vec();
        bool needs_reshape = false;
        for (size_t i = input_sizes.size() - 3; i < input_sizes.size(); i++) {
            if (input_sizes[i] < 1) {
                input_sizes[i] = 1;
                needs_reshape = true;
            }
        }
        if (needs_reshape) {
            int64_t total_elements = input.numel();
            int64_t new_total = 1;
            for (auto s : input_sizes) new_total *= s;
            if (new_total > total_elements) {
                // Pad with zeros if needed
                input = torch::zeros(input_sizes, input.options());
            } else {
                input = input.reshape(input_sizes);
            }
        }
        
        // Ensure input is floating point for pooling
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Parse output size parameters from the remaining data
        std::vector<int64_t> output_size;
        
        // Try to extract 3 values for D, H, W output sizes
        for (int i = 0; i < 3 && offset + 1 <= Size; i++) {
            // Use single byte for more efficient fuzzing
            int64_t size_val = static_cast<int64_t>(Data[offset++] % 16) + 1;  // 1-16
            output_size.push_back(size_val);
        }
        
        // If we couldn't extract all 3 values, fill with defaults
        while (output_size.size() < 3) {
            output_size.push_back(1);
        }
        
        // Ensure output sizes don't exceed input spatial sizes
        auto in_sizes = input.sizes();
        int spatial_offset = input.dim() - 3;
        for (int i = 0; i < 3; i++) {
            if (output_size[i] > in_sizes[spatial_offset + i]) {
                output_size[i] = in_sizes[spatial_offset + i];
            }
            // Ensure at least 1
            if (output_size[i] < 1) {
                output_size[i] = 1;
            }
        }
        
        // Create AdaptiveMaxPool3d module with different output size configurations
        torch::nn::AdaptiveMaxPool3d pool = nullptr;
        
        // Try different output size configurations
        uint8_t config_type = 0;
        if (offset < Size) {
            config_type = Data[offset++] % 3;
        }
        
        try {
            switch (config_type) {
                case 0:
                    // Single integer for all dimensions
                    pool = torch::nn::AdaptiveMaxPool3d(
                        torch::nn::AdaptiveMaxPool3dOptions(output_size[0]));
                    break;
                    
                case 1:
                    // Tuple of three integers
                    pool = torch::nn::AdaptiveMaxPool3d(
                        torch::nn::AdaptiveMaxPool3dOptions({output_size[0], output_size[1], output_size[2]}));
                    break;
                    
                case 2:
                    // Another variation with tuple
                    pool = torch::nn::AdaptiveMaxPool3d(
                        torch::nn::AdaptiveMaxPool3dOptions({output_size[2], output_size[1], output_size[0]}));
                    break;
            }
        } catch (const std::exception &e) {
            // Invalid configuration, use default
            pool = torch::nn::AdaptiveMaxPool3d(
                torch::nn::AdaptiveMaxPool3dOptions(1));
        }
        
        // Apply the pooling operation
        torch::Tensor output;
        
        // Try with return indices
        bool return_indices = false;
        if (offset < Size) {
            return_indices = Data[offset++] % 2 == 0;
        }
        
        if (return_indices) {
            torch::Tensor indices;
            std::tie(output, indices) = pool->forward_with_indices(input);
            
            // Verify indices tensor is valid
            auto idx_sizes = indices.sizes();
            (void)idx_sizes;
        } else {
            output = pool->forward(input);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        auto mean_val = output.mean();
        
        // Try to access the output tensor's properties
        auto sizes = output.sizes();
        auto dtype = output.dtype();
        (void)sizes;
        (void)dtype;
        (void)sum;
        (void)mean_val;
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
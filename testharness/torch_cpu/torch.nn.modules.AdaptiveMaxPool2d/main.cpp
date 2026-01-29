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
        
        // Need at least a few bytes to create a tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Get output size parameters from the input data first
        int64_t output_h = (Data[offset++] % 8) + 1; // 1-8
        int64_t output_w = (Data[offset++] % 8) + 1; // 1-8
        uint8_t config_type = Data[offset++] % 3;
        
        // Get spatial dimensions from fuzzer data
        int64_t input_h = (Data[offset++] % 16) + output_h; // Must be >= output_h
        int64_t input_w = (Data[offset++] % 16) + output_w; // Must be >= output_w
        int64_t channels = (Data[offset++] % 4) + 1; // 1-4 channels
        int64_t batch_size = (Data[offset++] % 3) + 1; // 1-3 batch size
        
        // Create input tensor with appropriate shape for AdaptiveMaxPool2d
        // AdaptiveMaxPool2d expects 3D (C, H, W) or 4D (N, C, H, W) input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input to valid 4D shape for AdaptiveMaxPool2d
        int64_t total_elements = input.numel();
        if (total_elements == 0) {
            total_elements = 1;
            input = torch::zeros({1});
        }
        
        // Calculate how many elements we need for the target shape
        int64_t needed_elements = batch_size * channels * input_h * input_w;
        
        // Resize or repeat to match the needed size
        if (total_elements < needed_elements) {
            // Repeat the tensor to get enough elements
            int64_t repeat_factor = (needed_elements / total_elements) + 1;
            input = input.flatten().repeat({repeat_factor});
        }
        input = input.flatten().slice(0, 0, needed_elements).reshape({batch_size, channels, input_h, input_w});
        
        // Convert to float for pooling operations
        input = input.to(torch::kFloat32);
        
        // Create AdaptiveMaxPool2d module with different output size configurations
        torch::nn::AdaptiveMaxPool2d pool = nullptr;
        
        try {
            switch (config_type) {
                case 0:
                    // Single integer output size (square output)
                    pool = torch::nn::AdaptiveMaxPool2d(
                        torch::nn::AdaptiveMaxPool2dOptions(output_h));
                    break;
                case 1:
                    // Tuple of two integers
                    pool = torch::nn::AdaptiveMaxPool2d(
                        torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
                    break;
                case 2:
                    // Also test with 3D input (C, H, W)
                    input = input.squeeze(0); // Remove batch dimension
                    pool = torch::nn::AdaptiveMaxPool2d(
                        torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
                    break;
            }
            
            // Apply the AdaptiveMaxPool2d operation
            auto output = pool->forward(input);
            
            // Use the output to ensure it's not optimized away
            if (output.numel() > 0) {
                volatile float sum_val = output.sum().item<float>();
                (void)sum_val;
            }
            
            // Also test with return_indices option if available
            // AdaptiveMaxPool2d with indices returns a tuple
            auto pool_with_indices = torch::nn::AdaptiveMaxPool2d(
                torch::nn::AdaptiveMaxPool2dOptions({output_h, output_w}));
            
            // Call forward_with_indices to get both output and indices
            auto result = pool_with_indices->forward_with_indices(
                config_type == 2 ? input.unsqueeze(0) : input);
            auto pooled_output = std::get<0>(result);
            auto indices = std::get<1>(result);
            
            // Verify indices are valid
            if (indices.numel() > 0) {
                volatile int64_t max_idx = indices.max().item<int64_t>();
                (void)max_idx;
            }
        }
        catch (const c10::Error&) {
            // Expected errors from invalid shapes/configurations - silently ignore
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
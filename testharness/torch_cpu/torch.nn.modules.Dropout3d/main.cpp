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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for Dropout3d first
        float p = static_cast<float>(Data[offset]) / 255.0f;
        offset++;
        
        bool inplace = Data[offset] % 2 == 1;
        offset++;
        
        bool training_mode = Data[offset] % 2 == 1;
        offset++;
        
        // Extract dimensions for 5D tensor (N, C, D, H, W)
        // Use small dimensions to avoid memory issues
        int64_t N = (Data[offset] % 4) + 1;  // 1-4
        offset++;
        int64_t C = (Data[offset] % 8) + 1;  // 1-8
        offset++;
        int64_t D = (Data[offset] % 4) + 1;  // 1-4
        offset++;
        int64_t H = (Data[offset] % 8) + 1;  // 1-8
        offset++;
        int64_t W = (Data[offset] % 8) + 1;  // 1-8
        offset++;
        
        // Create 5D input tensor for Dropout3d (expects (N, C, D, H, W) or (C, D, H, W))
        torch::Tensor input = torch::randn({N, C, D, H, W});
        
        // If we have remaining data, use it to initialize tensor values
        if (offset < Size) {
            torch::Tensor from_data = fuzzer_utils::createTensor(Data, Size, offset);
            // Use the fuzzer data to influence the input if shapes are compatible
            if (from_data.numel() > 0) {
                try {
                    // Flatten and take what we can use
                    auto flat_data = from_data.flatten();
                    auto flat_input = input.flatten();
                    int64_t copy_size = std::min(flat_data.numel(), flat_input.numel());
                    if (copy_size > 0) {
                        flat_input.slice(0, 0, copy_size).copy_(flat_data.slice(0, 0, copy_size));
                    }
                } catch (...) {
                    // Silently ignore shape incompatibility
                }
            }
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout_module(torch::nn::Dropout3dOptions().p(p).inplace(inplace));
        
        // Set the module's training mode
        dropout_module->train(training_mode);
        
        // Apply Dropout3d to the input tensor
        torch::Tensor output;
        if (inplace) {
            // For inplace operation, we need a clone to avoid modifying original
            torch::Tensor input_clone = input.clone();
            output = dropout_module->forward(input_clone);
        } else {
            output = dropout_module->forward(input);
        }
        
        // Verify output is not empty
        if (output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
            
            // Additional checks to exercise more code paths
            volatile auto mean_val = output.mean().item<float>();
            (void)mean_val;
        }
        
        // Also test with 4D input (C, D, H, W) - unbatched
        torch::Tensor input_4d = torch::randn({C, D, H, W});
        torch::Tensor output_4d = dropout_module->forward(input_4d);
        if (output_4d.numel() > 0) {
            volatile float sum_4d = output_4d.sum().item<float>();
            (void)sum_4d;
        }
        
        // Test eval mode (no dropout applied)
        dropout_module->eval();
        torch::Tensor output_eval = dropout_module->forward(input);
        if (output_eval.numel() > 0) {
            volatile float sum_eval = output_eval.sum().item<float>();
            (void)sum_eval;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
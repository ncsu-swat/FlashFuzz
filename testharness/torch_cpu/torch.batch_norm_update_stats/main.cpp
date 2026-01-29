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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - batch_norm expects at least 2D tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point (required for batch norm)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // batch_norm_update_stats requires at least 2D input (N, C, ...)
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0); // Make it (1, 1)
            } else {
                input = input.unsqueeze(0); // Make it (1, C)
            }
        }
        
        // Get number of features (channel dimension)
        int64_t num_features = input.size(1);
        if (num_features <= 0) {
            return 0;
        }
        
        // Determine if we should use running stats or None
        bool use_running_stats = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Get momentum value (between 0 and 1)
        double momentum = 0.1;
        if (offset < Size) {
            momentum = static_cast<double>(Data[offset++]) / 255.0;
            // Clamp momentum to reasonable range
            momentum = std::max(0.01, std::min(1.0, momentum));
        }
        
        torch::Tensor mean, invstd;
        
        if (use_running_stats) {
            // Create running_mean and running_var tensors
            // They must be 1D with size == num_features
            torch::Tensor running_mean = torch::zeros({num_features}, input.options());
            torch::Tensor running_var = torch::ones({num_features}, input.options());
            
            // Optionally initialize with random values from fuzzer
            if (offset + 1 < Size && Data[offset++] % 2 == 0) {
                // Initialize with some variation
                for (int64_t i = 0; i < num_features && offset < Size; i++) {
                    running_mean[i] = static_cast<float>(Data[offset++] % 256) / 128.0f - 1.0f;
                    if (offset < Size) {
                        running_var[i] = std::max(0.01f, static_cast<float>(Data[offset++] % 256) / 128.0f);
                    }
                }
            }
            
            // Call batch_norm_update_stats with running stats
            // Note: running_mean and running_var are updated IN-PLACE
            std::tie(mean, invstd) = torch::batch_norm_update_stats(
                input, running_mean, running_var, momentum);
            
            // Access running stats to ensure they were updated
            (void)running_mean.sum().item<float>();
            (void)running_var.sum().item<float>();
        } else {
            // Call with empty tensors (no running stats update)
            torch::Tensor empty_mean;
            torch::Tensor empty_var;
            
            std::tie(mean, invstd) = torch::batch_norm_update_stats(
                input, empty_mean, empty_var, momentum);
        }
        
        // Access results to ensure computation completed
        (void)mean.sum().item<float>();
        (void)invstd.sum().item<float>();
        
        // Additional test: verify output shapes
        if (mean.size(0) != num_features || invstd.size(0) != num_features) {
            std::cerr << "Unexpected output shape" << std::endl;
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors (shape mismatches, etc.) - expected during fuzzing
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
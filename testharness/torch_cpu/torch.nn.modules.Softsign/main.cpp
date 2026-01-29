#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <limits>         // For numeric_limits

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create Softsign module
        torch::nn::Softsign softsign_module;
        
        // Apply Softsign operation: x / (1 + |x|)
        torch::Tensor output = softsign_module->forward(input);
        
        // Test with different tensor properties
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply Softsign to the second tensor
            torch::Tensor output2 = softsign_module->forward(input2);
        }
        
        // Test with edge cases if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Only apply extreme scaling to floating point tensors
            if (extreme_input.is_floating_point()) {
                // Multiply by a large value to create potential overflow
                extreme_input = extreme_input * 1e10;
                
                // Apply Softsign to extreme values
                torch::Tensor extreme_output = softsign_module->forward(extreme_input);
            }
        }
        
        // Test with zero tensor
        if (input.numel() > 0) {
            try {
                torch::Tensor zero_input = torch::zeros_like(input);
                torch::Tensor zero_output = softsign_module->forward(zero_input);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test with very small values (only for floating point)
        if (input.is_floating_point() && input.numel() > 0) {
            try {
                torch::Tensor small_input = input * 1e-10;
                torch::Tensor small_output = softsign_module->forward(small_input);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test with NaN and Inf values if tensor is floating point
        if (input.is_floating_point() && input.numel() > 2) {
            try {
                torch::Tensor special_input = input.clone();
                
                // Flatten for easier indexing, then reshape back
                auto flat = special_input.flatten();
                flat[0] = std::numeric_limits<float>::quiet_NaN();
                flat[flat.numel() - 1] = std::numeric_limits<float>::infinity();
                special_input = flat.view(input.sizes());
                
                // Apply Softsign to special values
                torch::Tensor special_output = softsign_module->forward(special_input);
            } catch (...) {
                // Silently handle expected failures (e.g., for non-contiguous tensors)
            }
        }
        
        // Test with negative values
        if (input.is_floating_point() && input.numel() > 0) {
            try {
                torch::Tensor neg_input = -torch::abs(input) - 1.0;
                torch::Tensor neg_output = softsign_module->forward(neg_input);
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test batched input with multiple dimensions
        if (offset + 4 < Size && input.numel() > 0) {
            try {
                int batch_size = (Data[offset] % 4) + 1;
                offset++;
                torch::Tensor batched_input = input.unsqueeze(0).expand({batch_size, -1});
                if (batched_input.dim() > 0) {
                    batched_input = batched_input.contiguous();
                    torch::Tensor batched_output = softsign_module->forward(batched_input);
                }
            } catch (...) {
                // Silently handle shape-related failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
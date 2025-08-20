#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create grid tensor
        torch::Tensor grid;
        if (offset < Size) {
            grid = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible grid tensor
            if (input.dim() >= 4) {
                // For cudnn_grid_sampler, grid should be same batch size as input
                // and have shape [N, H_out, W_out, 2]
                int64_t N = input.size(0);
                int64_t H_out = input.size(2) > 0 ? input.size(2) : 1;
                int64_t W_out = input.size(3) > 0 ? input.size(3) : 1;
                grid = torch::zeros({N, H_out, W_out, 2}, input.options());
            } else {
                // Create a default grid for lower dimensional inputs
                grid = torch::zeros({1, 1, 1, 2}, input.options());
            }
        }
        
        // Try to move tensors to CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            grid = grid.cuda();
        }
        
        // Apply cudnn_grid_sampler
        // Note: cudnn_grid_sampler requires 4D input tensor and 4D grid tensor
        // If tensors don't have 4 dimensions, we'll let the operation handle the error
        torch::Tensor output;
        
        // Call the operation
        output = torch::cudnn_grid_sampler(input, grid);
        
        // Perform some operation on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
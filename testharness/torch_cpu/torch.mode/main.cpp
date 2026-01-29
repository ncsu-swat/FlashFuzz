#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with mode result

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
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to operate on
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get keepdim boolean
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply torch.mode operation
        // mode returns a tuple of (values, indices)
        if (input.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            int64_t ndim = input.dim();
            dim = ((dim % ndim) + ndim) % ndim;  // Proper modulo for negative values
            
            // Call mode operation
            auto result = torch::mode(input, dim, keepdim);
            
            // Access the values and indices from the result
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Perform some operations with the results to ensure they're used
            // Use sum() which works on any numeric dtype
            auto sum_values = values.sum();
            auto sum_indices = indices.sum();
            
            // Force computation without type-specific item() calls
            (void)sum_values.numel();
            (void)sum_indices.numel();
        } else {
            // For 0-dim tensors, mode requires at least 1-dim
            // Try calling with dim=-1 which should be handled by PyTorch
            try {
                auto result = torch::mode(input, /*dim=*/-1, /*keepdim=*/false);
                
                torch::Tensor values = std::get<0>(result);
                torch::Tensor indices = std::get<1>(result);
                
                (void)values.numel();
                (void)indices.numel();
            } catch (...) {
                // Expected to fail for 0-dim tensors, silently catch
            }
        }
        
        // Also test mode with different dimensions on multi-dim tensors
        if (input.dim() >= 2 && offset < Size) {
            int64_t alt_dim = static_cast<int64_t>(Data[offset] % input.dim());
            try {
                auto result = torch::mode(input, alt_dim, !keepdim);
                torch::Tensor values = std::get<0>(result);
                torch::Tensor indices = std::get<1>(result);
                (void)values.numel();
                (void)indices.numel();
            } catch (...) {
                // Silently catch expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
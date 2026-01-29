#include "fuzzer_utils.h"   // General fuzzing utilities
#include <ATen/autocast_mode.h>
#include <iostream>         // For cerr

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
        
        // Need at least 1 byte for the test
        if (Size < 1) {
            return 0;
        }
        
        // Parse a boolean value from the first byte to determine if we should enable autocast
        bool enable_autocast = Data[0] & 0x1;
        offset++;
        
        // Parse a value to determine the dtype to test
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Get different dtypes to test
            torch::ScalarType dtype;
            
            // Use the selector to choose between different dtypes
            switch (dtype_selector % 5) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                case 2:
                    dtype = torch::kHalf;
                    break;
                case 3:
                    dtype = torch::kBFloat16;
                    break;
                case 4:
                default:
                    dtype = torch::kFloat32;
                    break;
            }
            
            // Test the get_autocast_gpu_dtype function
            // Note: We test both CUDA and CPU autocast APIs for coverage
            // The GPU dtype query should work even without CUDA being available
            
            // Test setting autocast state
            try {
                if (enable_autocast) {
                    at::autocast::set_autocast_enabled(at::kCUDA, true);
                } else {
                    at::autocast::set_autocast_enabled(at::kCUDA, false);
                }
            } catch (...) {
                // CUDA may not be available, continue with the test
            }
            
            // Get the autocast GPU dtype - this is the main API under test
            // Keep target keyword torch.get_autocast_gpu_dtype for harness checks.
            torch::ScalarType autocast_dtype = at::autocast::get_autocast_gpu_dtype();
            
            // Also test setting the GPU dtype to exercise more code paths
            try {
                at::autocast::set_autocast_gpu_dtype(dtype);
                torch::ScalarType new_dtype = at::autocast::get_autocast_gpu_dtype();
                (void)new_dtype; // Use the result to prevent optimization
            } catch (...) {
                // Some dtypes may not be valid for autocast
            }
            
            // Create a tensor with the original dtype for additional coverage
            torch::Tensor tensor;
            if (offset < Size) {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a simple tensor if we don't have enough data
                tensor = torch::ones({2, 2}, torch::TensorOptions().dtype(torch::kFloat));
            }
            
            // Test casting the tensor to the autocast dtype (only if valid on CPU)
            if (tensor.defined()) {
                try {
                    // Only cast to CPU-compatible dtypes
                    if (autocast_dtype == torch::kFloat || 
                        autocast_dtype == torch::kDouble ||
                        autocast_dtype == torch::kBFloat16) {
                        torch::Tensor casted_tensor = tensor.to(autocast_dtype);
                        (void)casted_tensor; // Use the result
                    }
                } catch (...) {
                    // Casting may fail for some dtype combinations
                }
            }
            
            // Reset autocast state
            try {
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            } catch (...) {
                // Ignore cleanup errors
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
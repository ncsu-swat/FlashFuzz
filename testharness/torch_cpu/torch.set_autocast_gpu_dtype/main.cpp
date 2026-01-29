#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/autocast_mode.h>

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
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype to set for autocast
        // Autocast typically supports float16 and bfloat16
        uint8_t dtype_selector = Data[offset++];
        at::ScalarType autocast_dtype;
        
        // Limit to dtypes that make sense for autocast
        switch (dtype_selector % 4) {
            case 0:
                autocast_dtype = at::kHalf;
                break;
            case 1:
                autocast_dtype = at::kBFloat16;
                break;
            case 2:
                autocast_dtype = at::kFloat;
                break;
            default:
                autocast_dtype = at::kHalf;
                break;
        }
        
        // Set the autocast GPU dtype using the at::autocast namespace
        // This is the C++ equivalent of torch.set_autocast_gpu_dtype
        at::autocast::set_autocast_gpu_dtype(autocast_dtype);
        
        // Verify by getting the dtype back
        at::ScalarType retrieved_dtype = at::autocast::get_autocast_gpu_dtype();
        (void)retrieved_dtype; // Use the value to avoid compiler warnings
        
        // Create a tensor to test with
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (!tensor.defined()) {
                return 0;
            }
            
            // Ensure tensor is float type for meaningful autocast testing
            try {
                if (!tensor.is_floating_point()) {
                    tensor = tensor.to(torch::kFloat32);
                }
            } catch (...) {
                // Silently ignore conversion failures
                return 0;
            }
            
            // Test autocast enabled/disabled states
            bool enabled = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
            (void)enabled;
            
            // Test operations that would be affected by autocast on CPU
            // Note: Autocast primarily affects CUDA operations, but we can still
            // exercise the API and context management on CPU
            try {
                // Perform operations that autocast would affect
                torch::Tensor result1 = tensor + tensor;
                torch::Tensor result2 = torch::matmul(tensor.view({-1, 1}), tensor.view({1, -1}));
                (void)result1;
                (void)result2;
            } catch (...) {
                // Shape mismatches are expected, ignore silently
            }
            
            // Test changing the dtype multiple times
            if (offset < Size) {
                uint8_t new_dtype_selector = Data[offset++];
                at::ScalarType new_dtype;
                
                switch (new_dtype_selector % 3) {
                    case 0:
                        new_dtype = at::kHalf;
                        break;
                    case 1:
                        new_dtype = at::kBFloat16;
                        break;
                    default:
                        new_dtype = at::kFloat;
                        break;
                }
                
                // Change the autocast dtype
                at::autocast::set_autocast_gpu_dtype(new_dtype);
                
                // Verify the change
                at::ScalarType check_dtype = at::autocast::get_autocast_gpu_dtype();
                (void)check_dtype;
            }
            
            // Test setting dtype back to a known value (cleanup)
            at::autocast::set_autocast_gpu_dtype(at::kHalf);
        }
        
        // Also test autocast cache clearing
        at::autocast::clear_cache();
        
        // Test checking if autocast is enabled for different device types
        bool cuda_enabled = at::autocast::is_autocast_enabled(at::kCUDA);
        bool cpu_enabled = at::autocast::is_autocast_enabled(at::kCPU);
        (void)cuda_enabled;
        (void)cpu_enabled;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
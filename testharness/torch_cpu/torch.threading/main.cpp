#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/Parallel.h>  // For threading functions

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
        
        // Need at least a few bytes to create meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to work with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test get_num_threads
        int num_threads = at::get_num_threads();
        
        // Test set_num_threads with valid values (must be >= 1)
        if (offset < Size) {
            int new_thread_count = static_cast<int>(Data[offset++]) % 16 + 1; // 1-16 threads
            at::set_num_threads(new_thread_count);
            
            // Verify the change took effect
            int updated_threads = at::get_num_threads();
            (void)updated_threads;
            
            // Do some computation with the new thread count
            torch::Tensor result = tensor + 1;
            (void)result;
        }
        
        // Test get_num_interop_threads
        int num_interop_threads = at::get_num_interop_threads();
        (void)num_interop_threads;
        
        // Test set_num_interop_threads with valid values
        // Note: set_num_interop_threads can only be called once before any inter-op parallel work
        // So we skip changing it dynamically to avoid issues
        
        // Test in_parallel_region (ATen function)
        bool in_parallel = at::in_parallel_region();
        (void)in_parallel;
        
        // Test NoGradGuard
        if (offset < Size) {
            bool enable_grad = Data[offset++] % 2 == 0;
            
            if (enable_grad) {
                // Create a tensor that requires gradients
                torch::Tensor grad_tensor = torch::ones({3, 3}, torch::dtype(torch::kFloat32)).set_requires_grad(true);
                
                // Do some computation that would accumulate gradients
                torch::Tensor result = grad_tensor.pow(3);
                
                if (result.requires_grad()) {
                    // Sum to get a scalar for backward
                    torch::Tensor sum_result = result.sum();
                    sum_result.backward();
                }
            } else {
                // Use NoGradGuard to disable gradients
                torch::NoGradGuard no_grad;
                torch::Tensor grad_tensor = torch::ones({3, 3}, torch::dtype(torch::kFloat32)).set_requires_grad(true);
                torch::Tensor result = grad_tensor.pow(3);
                (void)result;
            }
        }
        
        // Test with different valid thread counts
        if (offset < Size) {
            std::vector<int> thread_counts = {1, 2, 4, 
                                             static_cast<int>(Data[offset++]) % 16 + 1};
            
            for (int count : thread_counts) {
                // Set valid thread count
                at::set_num_threads(count);
                
                // Do some computation with this thread count
                torch::Tensor result = tensor.sin();
                (void)result;
            }
        }
        
        // Test parallel_run and related APIs
        if (offset < Size && tensor.numel() > 0) {
            // Test get_thread_num (returns current thread number in parallel region)
            int thread_num = at::get_thread_num();
            (void)thread_num;
            
            // Perform operations that may use parallelism internally
            torch::Tensor large_tensor = torch::randn({100, 100});
            torch::Tensor mm_result = torch::mm(large_tensor, large_tensor.t());
            (void)mm_result;
            
            // Test with different thread settings
            int original_threads = at::get_num_threads();
            
            int new_threads = static_cast<int>(Data[offset++]) % 8 + 1;
            at::set_num_threads(new_threads);
            
            // Run computation that benefits from threading
            torch::Tensor conv_input = torch::randn({1, 3, 32, 32});
            torch::Tensor conv_weight = torch::randn({16, 3, 3, 3});
            torch::Tensor conv_result = torch::conv2d(conv_input, conv_weight);
            (void)conv_result;
            
            // Restore original thread count
            at::set_num_threads(original_threads);
        }
        
        // Test edge case: setting to 1 thread (single-threaded mode)
        at::set_num_threads(1);
        torch::Tensor single_thread_result = tensor.cos();
        (void)single_thread_result;
        
        // Restore reasonable thread count
        at::set_num_threads(std::max(1, num_threads));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
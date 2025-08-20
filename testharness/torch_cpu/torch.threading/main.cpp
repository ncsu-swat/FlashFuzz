#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <thread>         // For threading functionality

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        int num_threads = torch::get_num_threads();
        
        // Test set_num_threads with various values
        if (offset < Size) {
            int new_thread_count = static_cast<int>(Data[offset++]) % 16 + 1; // 1-16 threads
            torch::set_num_threads(new_thread_count);
            
            // Verify the change took effect
            int updated_threads = torch::get_num_threads();
            
            // Do some computation with the new thread count
            torch::Tensor result = tensor + 1;
        }
        
        // Test get_num_interop_threads
        int num_interop_threads = torch::get_num_interop_threads();
        
        // Test set_num_interop_threads
        if (offset < Size) {
            int new_interop_thread_count = static_cast<int>(Data[offset++]) % 8 + 1; // 1-8 threads
            torch::set_num_interop_threads(new_interop_thread_count);
            
            // Verify the change took effect
            int updated_interop_threads = torch::get_num_interop_threads();
            
            // Do some computation with the new interop thread count
            torch::Tensor result = tensor * 2;
        }
        
        // Test in_parallel_region
        bool in_parallel = torch::in_parallel_region();
        
        // Test NoGradGuard
        if (offset < Size) {
            bool enable_grad = Data[offset++] % 2 == 0;
            
            if (enable_grad) {
                // Create a tensor that requires gradients
                torch::Tensor grad_tensor = torch::ones_like(tensor, torch::requires_grad(true));
                
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
                torch::Tensor grad_tensor = torch::ones_like(tensor, torch::requires_grad(true));
                torch::Tensor result = grad_tensor.pow(3);
            }
        }
        
        // Test with different thread counts including edge cases
        if (offset < Size) {
            std::vector<int> thread_counts = {1, 0, -1, 
                                             static_cast<int>(Data[offset++]) % 32 + 1};
            
            for (int count : thread_counts) {
                // Try setting thread count (may throw for invalid values)
                try {
                    torch::set_num_threads(count);
                    
                    // Do some computation with this thread count
                    torch::Tensor result = tensor.sin();
                } catch (const std::exception& e) {
                    // Expected for invalid thread counts
                }
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
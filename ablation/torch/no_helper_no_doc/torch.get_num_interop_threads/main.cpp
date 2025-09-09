#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>  // PyTorch C++ frontend

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Test basic functionality - get_num_interop_threads doesn't take parameters
        // but we can still test it in various contexts
        
        // Basic call to get_num_interop_threads
        int num_threads = torch::get_num_interop_threads();
        
        // Verify the result is reasonable (should be >= 0)
        if (num_threads < 0) {
            std::cout << "Invalid number of interop threads: " << num_threads << std::endl;
        }
        
        // If we have enough data, use it to set interop threads and then get them
        if (Size >= sizeof(int)) {
            int new_num_threads;
            std::memcpy(&new_num_threads, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Clamp to reasonable range to avoid system issues
            new_num_threads = std::abs(new_num_threads) % 64 + 1; // 1-64 threads
            
            // Set the number of interop threads
            torch::set_num_interop_threads(new_num_threads);
            
            // Get the number again to verify it was set
            int retrieved_threads = torch::get_num_interop_threads();
            
            // The retrieved value should match what we set (or be system-limited)
            if (retrieved_threads <= 0) {
                std::cout << "Retrieved invalid interop threads after setting: " << retrieved_threads << std::endl;
            }
        }
        
        // Test multiple calls in sequence
        for (int i = 0; i < 5; i++) {
            int threads = torch::get_num_interop_threads();
            if (threads < 0) {
                std::cout << "Invalid threads in loop iteration " << i << ": " << threads << std::endl;
            }
        }
        
        // Test in different thread contexts if we have more data
        if (Size >= 2 * sizeof(int)) {
            int thread_count1, thread_count2;
            std::memcpy(&thread_count1, Data + offset, sizeof(int));
            offset += sizeof(int);
            std::memcpy(&thread_count2, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Clamp values
            thread_count1 = std::abs(thread_count1) % 32 + 1;
            thread_count2 = std::abs(thread_count2) % 32 + 1;
            
            // Set and get in sequence
            torch::set_num_interop_threads(thread_count1);
            int result1 = torch::get_num_interop_threads();
            
            torch::set_num_interop_threads(thread_count2);
            int result2 = torch::get_num_interop_threads();
            
            // Verify results are reasonable
            if (result1 <= 0 || result2 <= 0) {
                std::cout << "Invalid thread counts in sequence test: " << result1 << ", " << result2 << std::endl;
            }
        }
        
        // Final verification call
        int final_threads = torch::get_num_interop_threads();
        if (final_threads < 0) {
            std::cout << "Final thread count invalid: " << final_threads << std::endl;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
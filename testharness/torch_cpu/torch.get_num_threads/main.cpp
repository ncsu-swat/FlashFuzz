#include "fuzzer_utils.h"
#include <iostream>

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
        // Get the current number of threads (save for restoration)
        int original_threads = torch::get_num_threads();
        
        if (Size < 1) {
            // Even with no data, exercise the basic get function
            int current = torch::get_num_threads();
            (void)current;
            return 0;
        }
        
        size_t offset = 0;
        
        // Test setting various thread counts
        // Use fuzzer data to determine thread count, limited to valid range 1-128
        int new_thread_count = static_cast<int>(Data[offset] % 128) + 1;
        offset++;
        
        torch::set_num_threads(new_thread_count);
        int updated_threads = torch::get_num_threads();
        (void)updated_threads;
        
        // Create a tensor and perform operations that may utilize threads
        if (Size > offset + 4) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Operations that benefit from multi-threading
            try {
                torch::Tensor result1 = tensor.clone();
                torch::Tensor result2 = tensor + tensor;
                torch::Tensor result3 = torch::matmul(tensor.view({-1, 1}), tensor.view({1, -1}));
                (void)result1;
                (void)result2;
                (void)result3;
            } catch (...) {
                // Shape mismatches are expected, ignore silently
            }
        }
        
        // Test changing thread count mid-operation
        if (Size > offset) {
            int another_count = static_cast<int>(Data[offset] % 64) + 1;
            offset++;
            torch::set_num_threads(another_count);
            
            int check = torch::get_num_threads();
            (void)check;
        }
        
        // Test with thread count of 1 (single-threaded mode)
        torch::set_num_threads(1);
        int single_thread = torch::get_num_threads();
        (void)single_thread;
        
        // Test with a larger thread count
        if (Size > offset) {
            int large_count = static_cast<int>(Data[offset]) + 1;  // 1-256 range
            torch::set_num_threads(large_count);
            int large_result = torch::get_num_threads();
            (void)large_result;
        }
        
        // Restore original thread count to avoid affecting other tests
        torch::set_num_threads(original_threads);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
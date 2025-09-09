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
        
        // Need at least 4 bytes for an integer
        if (Size < 4) {
            return 0;
        }
        
        // Extract number of threads from fuzzer input
        int32_t num_threads;
        if (!extract_value(Data, Size, offset, num_threads)) {
            return 0;
        }
        
        // Store original number of threads to restore later
        int original_threads = torch::get_num_threads();
        
        // Test various edge cases and valid values
        std::vector<int> test_values;
        
        // Add the fuzzed value
        test_values.push_back(num_threads);
        
        // Add some edge cases
        test_values.push_back(0);      // Zero threads
        test_values.push_back(1);      // Single thread
        test_values.push_back(-1);     // Negative value
        test_values.push_back(-100);   // Large negative value
        test_values.push_back(1000);   // Large positive value
        test_values.push_back(INT32_MAX); // Maximum int value
        test_values.push_back(INT32_MIN); // Minimum int value
        
        // Test each value
        for (int threads : test_values) {
            try {
                // Set number of threads
                torch::set_num_threads(threads);
                
                // Verify the setting took effect (for valid values)
                int current_threads = torch::get_num_threads();
                
                // For valid positive values, check if setting was applied
                if (threads > 0) {
                    // The actual number of threads set might be clamped to system limits
                    // so we just verify it's positive
                    if (current_threads <= 0) {
                        std::cout << "Warning: set_num_threads(" << threads 
                                  << ") resulted in " << current_threads << " threads" << std::endl;
                    }
                }
                
                // Test some basic tensor operations to ensure threading still works
                auto tensor1 = torch::randn({10, 10});
                auto tensor2 = torch::randn({10, 10});
                auto result = torch::mm(tensor1, tensor2);
                
                // Verify result is valid
                if (!result.defined() || result.numel() != 100) {
                    std::cout << "Matrix multiplication failed with " << threads << " threads" << std::endl;
                }
                
            } catch (const std::exception& e) {
                // Some values might throw exceptions, which is acceptable behavior
                std::cout << "Exception with threads=" << threads << ": " << e.what() << std::endl;
            }
        }
        
        // Test rapid changes in thread count
        for (int i = 0; i < 10 && offset < Size - 4; ++i) {
            int32_t rapid_threads;
            if (extract_value(Data, Size, offset, rapid_threads)) {
                torch::set_num_threads(std::abs(rapid_threads % 32) + 1); // Keep in reasonable range
            }
        }
        
        // Test concurrent access if we have more data
        if (offset < Size - 8) {
            int32_t thread_count1, thread_count2;
            if (extract_value(Data, Size, offset, thread_count1) &&
                extract_value(Data, Size, offset, thread_count2)) {
                
                // Simulate potential race conditions
                torch::set_num_threads(std::abs(thread_count1 % 16) + 1);
                auto tensor = torch::randn({100, 100});
                torch::set_num_threads(std::abs(thread_count2 % 16) + 1);
                auto result = torch::sum(tensor);
                
                if (!result.defined()) {
                    std::cout << "Tensor operation failed during thread count changes" << std::endl;
                }
            }
        }
        
        // Restore original thread count
        torch::set_num_threads(original_threads);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
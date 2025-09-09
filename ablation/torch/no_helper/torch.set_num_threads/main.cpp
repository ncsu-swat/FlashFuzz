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
        
        // Need at least 4 bytes for an int
        if (Size < 4) {
            return 0;
        }
        
        // Extract number of threads from fuzzer input
        int num_threads;
        if (!extract_value(Data, Size, offset, num_threads)) {
            return 0;
        }
        
        // Store original number of threads to restore later
        int original_threads = torch::get_num_threads();
        
        // Test various edge cases and valid values
        std::vector<int> test_values = {num_threads};
        
        // Add some additional test cases based on the fuzzer input
        if (Size >= 8) {
            int additional_value;
            if (extract_value(Data, Size, offset, additional_value)) {
                test_values.push_back(additional_value);
            }
        }
        
        // Add some fixed edge cases
        test_values.insert(test_values.end(), {
            0,    // Zero threads
            1,    // Single thread
            -1,   // Negative value
            -100, // Large negative value
            2,    // Two threads
            4,    // Four threads
            8,    // Eight threads
            16,   // Sixteen threads
            32,   // Thirty-two threads
            64,   // Sixty-four threads
            128,  // Large number of threads
            1000, // Very large number
            INT_MAX, // Maximum integer value
            INT_MIN  // Minimum integer value
        });
        
        for (int threads : test_values) {
            try {
                // Test setting number of threads
                torch::set_num_threads(threads);
                
                // Verify the setting took effect (for valid positive values)
                if (threads > 0) {
                    int current_threads = torch::get_num_threads();
                    // Note: The actual number set might be clamped to system limits
                    // so we just verify it's positive
                    if (current_threads <= 0) {
                        std::cout << "Warning: set_num_threads(" << threads 
                                 << ") resulted in " << current_threads << " threads" << std::endl;
                    }
                }
                
                // Test with some basic tensor operations to ensure threading works
                if (Size >= 12) {
                    auto tensor1 = torch::randn({10, 10});
                    auto tensor2 = torch::randn({10, 10});
                    auto result = torch::mm(tensor1, tensor2);
                    
                    // Force computation
                    result.sum().item<float>();
                }
                
                // Test multiple consecutive calls
                torch::set_num_threads(threads);
                torch::set_num_threads(threads);
                
            } catch (const std::exception& inner_e) {
                // Some thread values might throw exceptions, which is acceptable
                std::cout << "Inner exception for threads=" << threads 
                         << ": " << inner_e.what() << std::endl;
            }
        }
        
        // Test rapid changes in thread count
        if (Size >= 16) {
            for (int i = 0; i < 10; ++i) {
                int thread_val = (num_threads + i) % 32 + 1; // Keep positive
                torch::set_num_threads(thread_val);
            }
        }
        
        // Test setting threads before and after tensor operations
        torch::set_num_threads(std::max(1, abs(num_threads % 16) + 1));
        auto test_tensor = torch::ones({5, 5});
        torch::set_num_threads(std::max(1, abs(num_threads % 8) + 1));
        test_tensor = test_tensor * 2.0;
        
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
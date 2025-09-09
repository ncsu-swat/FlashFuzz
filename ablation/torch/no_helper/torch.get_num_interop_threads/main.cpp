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

        // Test basic functionality - get_num_interop_threads should always work
        int num_threads = torch::get_num_interop_threads();
        
        // Validate that the returned value is reasonable
        if (num_threads < 0) {
            std::cout << "Invalid negative thread count: " << num_threads << std::endl;
            return -1;
        }
        
        // Test multiple calls to ensure consistency and no crashes
        for (int i = 0; i < 10; ++i) {
            int threads_check = torch::get_num_interop_threads();
            if (threads_check != num_threads) {
                std::cout << "Thread count changed between calls: " << num_threads 
                         << " vs " << threads_check << std::endl;
            }
        }
        
        // If we have input data, use it to stress test by calling the function
        // in a loop based on the input size
        if (Size > 0) {
            size_t iterations = (Size % 100) + 1; // 1 to 100 iterations
            for (size_t i = 0; i < iterations; ++i) {
                int result = torch::get_num_interop_threads();
                // Ensure result is still valid
                if (result < 0) {
                    std::cout << "Invalid thread count in iteration " << i 
                             << ": " << result << std::endl;
                    return -1;
                }
            }
        }
        
        // Test concurrent access if we have enough data
        if (Size >= 4) {
            // Use input data to determine number of concurrent calls
            uint32_t concurrent_calls = *reinterpret_cast<const uint32_t*>(Data) % 50 + 1;
            
            std::vector<std::thread> threads;
            std::vector<int> results(concurrent_calls);
            
            for (uint32_t i = 0; i < concurrent_calls; ++i) {
                threads.emplace_back([&results, i]() {
                    results[i] = torch::get_num_interop_threads();
                });
            }
            
            for (auto& t : threads) {
                t.join();
            }
            
            // Verify all results are consistent and valid
            for (uint32_t i = 0; i < concurrent_calls; ++i) {
                if (results[i] < 0) {
                    std::cout << "Invalid thread count from concurrent call " << i 
                             << ": " << results[i] << std::endl;
                    return -1;
                }
                if (results[i] != num_threads) {
                    std::cout << "Inconsistent thread count from concurrent call " << i 
                             << ": expected " << num_threads << ", got " << results[i] << std::endl;
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}
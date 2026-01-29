#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the number of threads
        if (Size < 1) {
            return 0;
        }
        
        // Extract number of threads from the input data
        int num_threads = 1;
        if (Size >= sizeof(int32_t)) {
            int32_t raw_threads;
            std::memcpy(&raw_threads, Data, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Sanitize to a reasonable range (1 to 128 threads)
            num_threads = 1 + (std::abs(raw_threads) % 128);
        } else {
            // Use the byte value, ensure at least 1 thread
            num_threads = 1 + (Data[0] % 128);
            offset += 1;
        }
        
        // Set the number of threads using set_num_threads
        torch::set_num_threads(num_threads);
        
        // Call the init_num_threads function (no parameters)
        // This function initializes internal thread pool state
        torch::init_num_threads();
        
        // Verify the setting worked by getting the current number of threads
        int current_threads = torch::get_num_threads();
        
        // Perform operations that might use multiple threads
        if (offset < Size) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform operations that can benefit from multi-threading
            auto result = tensor + 1;
            auto product = tensor * 2;
            
            // Matrix operations that use threading
            if (tensor.dim() >= 2 && tensor.size(0) > 0 && tensor.size(1) > 0) {
                try {
                    auto matmul_result = torch::mm(tensor.view({tensor.size(0), -1}), 
                                                    tensor.view({-1, tensor.size(0)}));
                } catch (...) {
                    // Shape mismatch is expected for some inputs
                }
            }
            
            // Reduction operations
            auto sum = result.sum();
            auto mean = result.mean();
            
            // Use the results to prevent optimization
            (void)sum;
            (void)mean;
            (void)current_threads;
        }
        
        // Also test with different thread counts in the same run
        if (offset + 1 < Size) {
            int second_num_threads = 1 + (Data[offset] % 64);
            torch::set_num_threads(second_num_threads);
            torch::init_num_threads();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
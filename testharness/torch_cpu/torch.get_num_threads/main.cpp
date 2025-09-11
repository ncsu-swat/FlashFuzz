#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Get the current number of threads
        int num_threads = torch::get_num_threads();
        
        // Try setting the number of threads to different values
        if (Size > 0) {
            int new_thread_count = static_cast<int>(Data[offset]) % 16 + 1;  // Limit to reasonable range 1-16
            torch::set_num_threads(new_thread_count);
            
            // Verify the setting worked
            int updated_threads = torch::get_num_threads();
            
            // Create a tensor and perform an operation to see if thread count affects behavior
            if (Size > offset + 1) {
                offset++;
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform some operation that might use multiple threads
                torch::Tensor result = tensor.clone();
                
                // Try a few more thread-related operations
                if (Size > offset + 1) {
                    int another_thread_count = static_cast<int>(Data[offset]) % 32 + 1;
                    torch::set_num_threads(another_thread_count);
                    
                    // Get the number of threads again
                    int final_threads = torch::get_num_threads();
                    
                    // Try with 0 threads (edge case)
                    torch::set_num_threads(0);
                    int zero_threads = torch::get_num_threads();
                    
                    // Try with negative threads (edge case)
                    if (Size > offset + 1) {
                        offset++;
                        int negative_threads = -static_cast<int>(Data[offset]);
                        torch::set_num_threads(negative_threads);
                        int after_negative = torch::get_num_threads();
                    }
                }
            }
            
            // Restore original thread count
            torch::set_num_threads(num_threads);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

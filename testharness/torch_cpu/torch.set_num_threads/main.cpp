#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the number of threads
        if (Size < 1) {
            return 0;
        }
        
        // Extract number of threads from the first byte
        int num_threads = static_cast<int>(Data[0]);
        offset++;
        
        // Try setting the number of threads
        torch::set_num_threads(num_threads);
        
        // Verify the setting took effect
        int current_threads = torch::get_num_threads();
        
        // Create a tensor and perform a simple operation to test the thread setting
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform a computation that could potentially use multiple threads
            torch::Tensor result = tensor.sum();
            
            // Try another operation
            if (tensor.dim() > 0) {
                torch::Tensor result2 = tensor.mean(0);
            }
        }
        
        // Try setting back to 1 thread
        torch::set_num_threads(1);
        
        // Try setting to a negative number of threads
        if (Size > offset && Data[offset] % 2 == 0) {
            torch::set_num_threads(-1 * (Data[offset] % 100));
            offset++;
        }
        
        // Try setting to a very large number of threads
        if (Size > offset) {
            int large_num = static_cast<int>(Data[offset]) * 1000;
            torch::set_num_threads(large_num);
            offset++;
        }
        
        // Try setting to zero threads
        torch::set_num_threads(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
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
        
        // Need at least 1 byte for the number of threads
        if (Size < 1) {
            return 0;
        }
        
        // Extract number of threads from the input data
        int64_t num_threads = 1;
        if (Size >= sizeof(int64_t)) {
            std::memcpy(&num_threads, Data, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            num_threads = static_cast<int64_t>(Data[0]);
            offset += 1;
        }
        
        // Set the number of threads using set_num_threads
        torch::set_num_threads(static_cast<int>(num_threads));
        
        // Call the init_num_threads function (no parameters)
        torch::init_num_threads();
        
        // Verify the setting worked by creating a tensor and performing an operation
        if (offset < Size) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform a simple operation that might use multiple threads
            auto result = tensor + 1;
            
            // Get the current number of threads to verify
            int current_threads = torch::get_num_threads();
            
            // Use the result and current_threads to prevent optimization
            if (result.numel() > 0 && current_threads > 0) {
                auto sum = result.sum();
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

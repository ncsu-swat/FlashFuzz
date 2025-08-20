#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the number of interop threads
        int num_threads = torch::get_num_interop_threads();
        
        // Try setting the number of interop threads to different values
        if (Size > 0) {
            int new_threads = static_cast<int>(Data[offset++]) % 16;
            torch::set_num_interop_threads(new_threads);
            
            // Verify the change took effect
            int updated_threads = torch::get_num_interop_threads();
            
            // Try setting to negative value (should be handled by PyTorch)
            if (Size > offset) {
                int negative_threads = -static_cast<int>(Data[offset++]);
                torch::set_num_interop_threads(negative_threads);
                
                // Get the value again after attempting to set negative
                int after_negative = torch::get_num_interop_threads();
            }
            
            // Try setting to a very large value
            if (Size > offset + sizeof(int)) {
                int large_value;
                std::memcpy(&large_value, Data + offset, sizeof(int));
                offset += sizeof(int);
                
                // Make it a large positive value
                large_value = std::abs(large_value);
                if (large_value < 1000) {
                    large_value += 1000;
                }
                
                torch::set_num_interop_threads(large_value);
                int after_large = torch::get_num_interop_threads();
            }
            
            // Try setting back to original value
            torch::set_num_interop_threads(num_threads);
            int restored_threads = torch::get_num_interop_threads();
        }
        
        // Create a tensor and perform an operation to see if thread settings affect it
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operation that might use interop threads
            torch::Tensor result = tensor + 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
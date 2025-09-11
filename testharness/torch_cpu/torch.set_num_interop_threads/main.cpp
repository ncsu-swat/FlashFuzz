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
        
        // Extract number of threads from the first byte
        int num_threads = static_cast<int>(Data[0]);
        offset++;
        
        // Try setting the number of interop threads
        torch::set_num_interop_threads(num_threads);
        
        // Verify the setting worked by getting the current value
        int current_threads = torch::get_num_interop_threads();
        
        // Create a tensor and perform some operation to test the threading
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operation that might use interop threads
            auto result = tensor.clone();
            
            // Use the result to prevent optimization
            if (result.defined()) {
                volatile float sum = result.sum().item<float>();
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

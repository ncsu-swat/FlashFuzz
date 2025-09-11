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
        
        // Need at least 1 byte for the mode
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine whether to enable or disable flush denormal
        bool mode = Data[offset++] & 0x1;
        
        // Set flush denormal mode
        torch::set_flush_denormal(mode);
        
        // Create a tensor with potentially denormal values to test the effect
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might generate denormal values
            torch::Tensor result = tensor * 1e-30;
            
            // Perform another operation to ensure the tensor is used
            torch::Tensor sum = result.sum();
            
            // Access the value to ensure the computation is performed
            float value = sum.item<float>();
            
            // Reset flush denormal mode to default (false)
            torch::set_flush_denormal(false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

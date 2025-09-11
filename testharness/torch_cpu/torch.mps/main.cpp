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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test if MPS is available
        bool is_mps_available = torch::mps::is_available();
        
        // Only proceed with MPS operations if it's available
        if (is_mps_available) {
            // Create MPS device
            auto mps_device = torch::Device(torch::kMPS);
            
            // Move tensor to MPS device
            torch::Tensor mps_tensor = tensor.to(mps_device);
            
            // Perform some operations on the MPS tensor
            torch::Tensor result = mps_tensor + 1;
            
            // Move back to CPU for verification
            torch::Tensor cpu_result = result.to(torch::kCPU);
            
            // Test synchronization
            torch::mps::synchronize();
            
            // Test manual seed
            torch::mps::manual_seed(42);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

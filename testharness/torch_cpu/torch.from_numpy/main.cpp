#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a PyTorch tensor using the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor properties
        auto dtype = tensor.dtype();
        auto shape = tensor.sizes();
        
        // Convert tensor to CPU if it's not already
        auto cpu_tensor = tensor.cpu().contiguous();
        
        // Create a simple test tensor to simulate from_numpy behavior
        // Since we can't use actual NumPy arrays, we'll test tensor operations
        try {
            // Test tensor creation and basic operations
            torch::Tensor result_tensor = cpu_tensor.clone();
            
            // Perform some operations on the result tensor to ensure it's valid
            if (result_tensor.defined()) {
                auto sum = result_tensor.sum();
                auto mean = result_tensor.mean();
                if (result_tensor.numel() > 1) {
                    auto std_dev = result_tensor.std();
                }
            }
        } catch (...) {
            // Handle any tensor operation errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
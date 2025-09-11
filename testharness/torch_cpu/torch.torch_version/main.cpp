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
        
        // Get the PyTorch version
        std::string version = TORCH_VERSION;
        
        // Try to create a tensor if there's enough data
        if (Size > 2) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test torch_version with a tensor in scope
            std::string version_again = TORCH_VERSION;
            
            // Use the tensor to prevent it from being optimized out
            if (tensor.defined()) {
                auto tensor_sum = tensor.sum();
            }
        }
        
        // Test torch_version with different contexts
        {
            // In a nested scope
            std::string version_nested = TORCH_VERSION;
        }
        
        // Test torch_version after some tensor operations
        if (Size > 4) {
            torch::Tensor t1 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (t1.defined() && t2.defined()) {
                try {
                    auto result = t1 + t2;
                } catch (...) {
                    // Ignore errors from tensor operations
                }
            }
            
            // Get version after tensor operations
            std::string version_after_ops = TORCH_VERSION;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

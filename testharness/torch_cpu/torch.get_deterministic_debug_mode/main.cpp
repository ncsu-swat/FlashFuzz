#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the current deterministic debug mode
        auto current_mode = at::globalContext().deterministicAlgorithms();
        
        // Try to create a tensor to test with deterministic operations
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test different deterministic debug modes
            if (offset < Size) {
                uint8_t mode_selector = Data[offset++] % 2;
                
                // Set deterministic debug mode based on the input
                switch (mode_selector) {
                    case 0:
                        at::globalContext().setDeterministicAlgorithms(true);
                        break;
                    case 1:
                        at::globalContext().setDeterministicAlgorithms(false);
                        break;
                }
                
                // Get the mode after setting it
                auto new_mode = at::globalContext().deterministicAlgorithms();
                
                // Perform some operation that might be affected by deterministic mode
                if (tensor.defined() && tensor.numel() > 0) {
                    try {
                        auto result = torch::matmul(tensor, tensor);
                    } catch (...) {
                        // Ignore exceptions from the operation itself
                    }
                }
            }
            
            // Reset to the original mode
            at::globalContext().setDeterministicAlgorithms(current_mode);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
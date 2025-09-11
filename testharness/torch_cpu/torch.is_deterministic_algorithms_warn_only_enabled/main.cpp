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
        
        // Get a boolean value from the input data to toggle deterministic algorithms state
        bool enable_warn_only = false;
        if (offset < Size) {
            enable_warn_only = Data[offset++] & 0x1;
        }
        
        // Set the deterministic algorithms warn-only mode
        at::globalContext().setDeterministicAlgorithmsWarnOnly(enable_warn_only);
        
        // Check if the warn-only mode is enabled
        bool is_warn_only_enabled = at::globalContext().deterministicAlgorithmsWarnOnly();
        
        // Verify that the setting was applied correctly
        if (is_warn_only_enabled != enable_warn_only) {
            throw std::runtime_error("Deterministic algorithms warn-only mode was not set correctly");
        }
        
        // Create a tensor to test with deterministic operations
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Enable deterministic algorithms
            at::globalContext().setDeterministicAlgorithms(true);
            
            // Perform an operation that requires deterministic algorithms
            // This will either warn or error depending on warn_only setting
            try {
                auto result = torch::nn::functional::max_pool2d(
                    tensor.reshape({1, 1, tensor.numel(), 1}), 
                    torch::nn::functional::MaxPool2dFuncOptions(2).stride(2)
                );
            } catch (const c10::Error& e) {
                // This is expected if warn_only is false and tensor isn't compatible
                // with deterministic operations
            }
            
            // Reset deterministic mode to avoid affecting other tests
            at::globalContext().setDeterministicAlgorithms(false);
        }
        
        // Toggle the warn-only mode again to test the transition
        if (offset < Size) {
            enable_warn_only = !enable_warn_only;
            at::globalContext().setDeterministicAlgorithmsWarnOnly(enable_warn_only);
            bool new_warn_only_enabled = at::globalContext().deterministicAlgorithmsWarnOnly();
            
            if (new_warn_only_enabled != enable_warn_only) {
                throw std::runtime_error("Failed to toggle deterministic algorithms warn-only mode");
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

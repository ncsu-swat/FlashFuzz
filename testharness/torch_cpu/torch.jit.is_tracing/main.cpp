#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we're currently tracing
        bool is_tracing = torch::jit::is_tracing();
        
        // Try to use the tensor in a simple operation to see if it affects tracing state
        torch::Tensor result = tensor + 1;
        
        // Check again if we're tracing
        bool is_tracing_after = torch::jit::is_tracing();
        
        // Verify that the tracing state is consistent
        if (is_tracing != is_tracing_after) {
            throw std::runtime_error("Tracing state changed unexpectedly");
        }
        
        // Try to create a simple script module and check tracing
        torch::jit::script::Module module;
        try {
            // Create a simple function that uses is_tracing
            auto checkTracingFunc = [&tensor]() -> bool {
                return torch::jit::is_tracing();
            };
            
            // Call the function to see if it works
            bool tracing_result = checkTracingFunc();
        } catch (...) {
            // Ignore exceptions from script module creation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

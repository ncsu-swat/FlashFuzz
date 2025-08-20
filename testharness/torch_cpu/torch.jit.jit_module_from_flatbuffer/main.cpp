#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a buffer from the input data
        std::vector<char> flatbuffer_data;
        flatbuffer_data.reserve(Size);
        for (size_t i = 0; i < Size; ++i) {
            flatbuffer_data.push_back(static_cast<char>(Data[i]));
        }
        
        // Try to load the module from the flatbuffer
        try {
            torch::jit::Module module = torch::jit::load_jit_module_from_flatbuffer(flatbuffer_data);
            
            // Create a tensor to pass to the module
            torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to get methods
            auto methods = module.get_methods();
            
            // If there are methods, try to run the first one
            if (!methods.empty()) {
                try {
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input);
                    auto output = methods[0](inputs);
                } catch (...) {
                    // Method execution might fail, which is expected for random data
                }
            }
        } catch (...) {
            // Expected to fail for most random inputs
        }
        
        // Try with a modified buffer (flip some bits)
        if (Size > 10) {
            std::vector<char> modified_buffer = flatbuffer_data;
            for (size_t i = 0; i < std::min(size_t(10), Size); ++i) {
                modified_buffer[i] ^= 0xFF;  // Flip bits
            }
            
            try {
                torch::jit::Module module = torch::jit::load_jit_module_from_flatbuffer(modified_buffer);
            } catch (...) {
                // Expected to fail
            }
        }
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
    return 0; // keep the input
}
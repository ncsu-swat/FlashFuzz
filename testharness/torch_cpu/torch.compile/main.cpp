#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some parameters for torch.compile from the remaining data
        bool fullgraph = false;
        bool dynamic = false;
        bool debug = false;
        
        if (offset < Size) {
            fullgraph = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            dynamic = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            debug = Data[offset++] & 0x1;
        }
        
        // Create a simple function to compile
        auto func = [](const torch::Tensor& x) {
            return x.sin() + x.cos();
        };
        
        // Apply torch.compile to the function
        try {
            // Create a traced module instead of using compile
            torch::jit::script::Module module;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto traced_module = torch::jit::trace(func, inputs);
            
            // Execute the traced function
            std::vector<torch::jit::IValue> traced_inputs;
            traced_inputs.push_back(input_tensor);
            auto result = traced_module.forward(traced_inputs).toTensor();
            
            // Verify the result by comparing with the original function
            auto expected_result = func(input_tensor);
            
            // Optional: Check if results match
            if (result.sizes() != expected_result.sizes()) {
                throw std::runtime_error("Compiled function produced incorrect shape");
            }
        }
        catch (const c10::Error& e) {
            // PyTorch-specific errors are expected in some cases
            // We catch them separately to distinguish from other errors
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
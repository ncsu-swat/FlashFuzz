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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input for the module
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a buffer for the IR module
        // Use the remaining data as the IR module buffer
        if (offset >= Size) {
            return 0;
        }
        
        size_t buffer_size = Size - offset;
        const char* buffer_ptr = reinterpret_cast<const char*>(Data + offset);
        
        // Try to import the IR module from the buffer
        try {
            std::string buffer_str(buffer_ptr, buffer_size);
            std::istringstream buffer_stream(buffer_str);
            torch::jit::Module module = torch::jit::load(buffer_stream);
            
            // Try to run the module with our input tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Attempt to forward the tensor through the module
            try {
                torch::jit::IValue output = module.forward(inputs);
                
                // Try to extract tensor from output
                if (output.isTensor()) {
                    torch::Tensor output_tensor = output.toTensor();
                }
            } catch (...) {
                // Forward might fail if the module expects different inputs
                // That's fine for fuzzing purposes
            }
        } catch (...) {
            // Import might fail if the buffer doesn't contain valid IR
            // That's expected for fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h"                         // General fuzzing utilities
#include <algorithm>                             // For std::min
#include <iostream>                              // For cerr
#include <memory>                                // For std::shared_ptr
#include <sstream>                               // For std::istringstream
#include <vector>                                // For std::vector
#include <torch/script.h>                        // For torch::jit::IValue
#include <torch/csrc/jit/serialization/import.h> // For torch::jit::import_ir_module
// Target API: torch.import_ir_module_from_buffer

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
        
        // Use the remaining data as the IR module buffer for torch.import_ir_module_from_buffer
        size_t remaining = Size - offset;
        size_t buffer_size = std::min(static_cast<size_t>(2048), remaining);
        std::string buffer_str;
        if (buffer_size > 0) {
            buffer_str.assign(reinterpret_cast<const char*>(Data + offset), buffer_size);
            offset += buffer_size;
        } else {
            buffer_str = "JITVERSION 4\n\n";
        }

        // Try to import the IR module from the buffer
        try {
            auto cu = std::make_shared<torch::jit::CompilationUnit>();
            std::istringstream buffer_stream(buffer_str);
            torch::jit::Module module = torch::jit::import_ir_module(cu, buffer_stream);
            
            // Try to run the module with our input tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Attempt to forward the tensor through the module
            try {
                torch::jit::IValue output = module.forward(inputs);
                
                // Try to extract tensor from output
                if (output.isTensor()) {
                    torch::Tensor output_tensor = output.toTensor();
                    (void)output_tensor.sum(); // Touch output to exercise execution
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

#include "fuzzer_utils.h"                         // General fuzzing utilities
#include <iostream>                               // For cerr, cout
#include <sstream>                                // For std::istringstream
#include <torch/script.h>                         // For torch::jit::Module
#include <torch/csrc/jit/serialization/import.h>  // For torch::jit::import_ir_module

// Target API: torch.import_ir_module

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        // Use fuzz data directly as potential IR/serialized module content
        // This tests the parser's robustness against malformed input
        std::string ir_content(reinterpret_cast<const char *>(Data), Size);
        
        // Try importing the IR module from the fuzz data buffer
        try
        {
            auto cu = std::make_shared<torch::jit::CompilationUnit>();
            std::istringstream ir_stream(ir_content);
            
            // Main API under test: import_ir_module
            torch::jit::Module module = torch::jit::import_ir_module(cu, ir_stream);

            // If import succeeds (very unlikely with random data), try to use the module
            try
            {
                // Get method names to see what's available
                auto methods = module.get_methods();
                
                // Try forward if it exists
                if (module.find_method("forward")) {
                    // Create a simple input tensor
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(torch::randn({1, 3, 32, 32}));
                    
                    torch::jit::IValue output = module.forward(inputs);
                    if (output.isTensor())
                    {
                        auto out_tensor = output.toTensor();
                        (void)out_tensor.sum(); // Touch output to exercise execution
                    }
                }
            }
            catch (...)
            {
                // Silently catch exceptions from running the module
                // Expected since fuzz data won't produce valid callable modules
            }
        }
        catch (...)
        {
            // Silently catch exceptions from importing the module
            // This is expected - most fuzz inputs won't be valid serialized modules
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
#include "fuzzer_utils.h"                         // General fuzzing utilities
#include <iostream>                               // For cerr
#include <sstream>                                // For std::istringstream
#include <torch/script.h>                         // For torch::jit::IValue
#include <torch/csrc/jit/serialization/import.h>  // For torch::jit::import_ir_module
// Target API: torch.import_ir_module

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
        
        // Create a tensor to use as input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Extract a bounded string from the remaining data to use as IR module content
        std::string ir_content;
        if (offset < Size)
        {
            size_t content_length = std::min(Size - offset, static_cast<size_t>(2048));
            ir_content = std::string(reinterpret_cast<const char *>(Data + offset), content_length);
            offset += content_length;
        }
        else
        {
            ir_content = "JITVERSION 4\n\n"; // Minimal header-like content to keep importer busy
        }

        // Try importing the IR module directly from the buffer with torch.import_ir_module
        try
        {
            auto cu = std::make_shared<torch::jit::CompilationUnit>();
            std::istringstream ir_stream(ir_content);
            torch::jit::Module module = torch::jit::import_ir_module(cu, ir_stream);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            try
            {
                torch::jit::IValue output = module.forward(inputs);
                if (output.isTensor())
                {
                    auto out_tensor = output.toTensor();
                    (void)out_tensor.sum(); // Touch output to exercise execution
                }
            }
            catch (...)
            {
                // Silently catch exceptions from running the module
            }
        }
        catch (...)
        {
            // Silently catch exceptions from importing the module
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

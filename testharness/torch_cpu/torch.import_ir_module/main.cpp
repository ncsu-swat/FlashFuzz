#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <fstream>        // For ofstream
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h> // For torch::jit::load

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a string from the remaining data to use as IR module content
        std::string ir_content;
        if (offset < Size) {
            size_t content_length = std::min(Size - offset, static_cast<size_t>(1024));
            ir_content = std::string(reinterpret_cast<const char*>(Data + offset), content_length);
            offset += content_length;
        } else {
            ir_content = "module {}";
        }
        
        // Create a temporary file to store the IR content
        std::string temp_filename = "temp_ir_module.ir";
        {
            std::ofstream temp_file(temp_filename);
            if (temp_file.is_open()) {
                temp_file << ir_content;
                temp_file.close();
            }
        }
        
        // Try to load the IR module using torch::jit::load
        try {
            auto module = torch::jit::load(temp_filename);
            
            // If we have a valid module, try to run it with our input tensor
            if (offset < Size && module) {
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Try to run the module with our input
                try {
                    auto output = module.forward(inputs);
                } catch (...) {
                    // Silently catch exceptions from running the module
                }
            }
        } catch (...) {
            // Silently catch exceptions from importing the module
        }
        
        // Clean up the temporary file
        std::remove(temp_filename.c_str());
        
        // Try creating a module from string using torch::jit::compile
        try {
            auto module = torch::jit::compile(ir_content);
            
            // If we have a valid module, try to run it with our input tensor
            if (offset < Size && module) {
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                // Try to run the module with our input
                try {
                    auto output = module.forward(inputs);
                } catch (...) {
                    // Silently catch exceptions from running the module
                }
            }
        } catch (...) {
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
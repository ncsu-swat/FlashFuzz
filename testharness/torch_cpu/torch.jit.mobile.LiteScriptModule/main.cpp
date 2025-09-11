#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <fstream>
#include <sstream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use as input for our model
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model file from the remaining data
        std::vector<uint8_t> model_data;
        if (offset < Size) {
            model_data.assign(Data + offset, Data + Size);
        } else {
            // If we've consumed all data, create a minimal valid model
            // This is a placeholder - in real fuzzing we'd want to generate valid model bytes
            return 0;
        }
        
        // Write model data to a temporary file
        std::string temp_filename = "temp_model.ptl";
        {
            std::ofstream model_file(temp_filename, std::ios::binary);
            if (!model_file) {
                return 0;
            }
            model_file.write(reinterpret_cast<const char*>(model_data.data()), model_data.size());
        }
        
        try {
            // Load the model using LiteScriptModule
            auto module = torch::jit::mobile::load_mobile_module_from_file(temp_filename);
            
            // Try to run the model with our input tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Run forward pass
            auto output = module.forward(inputs);
            
            // Try to access module methods
            auto method_names = module.get_method_names();
            
            // Try to run other methods if available
            for (const auto& method_name : method_names) {
                if (method_name != "forward") {
                    try {
                        module.run_method(method_name, inputs);
                    } catch (...) {
                        // Ignore exceptions from individual methods
                    }
                }
            }
            
            // Try to get model attributes
            auto attr_names = module.get_attributes();
            for (const auto& attr_name : attr_names) {
                try {
                    module.get_attribute(attr_name);
                } catch (...) {
                    // Ignore exceptions from individual attributes
                }
            }
            
            // Try to get buffers
            auto buffer_names = module.get_buffers();
            for (const auto& buffer_name : buffer_names) {
                try {
                    module.get_buffer(buffer_name);
                } catch (...) {
                    // Ignore exceptions from individual buffers
                }
            }
            
            // Try to get parameters
            auto param_names = module.get_parameters();
            for (const auto& param_name : param_names) {
                try {
                    module.get_parameter(param_name);
                } catch (...) {
                    // Ignore exceptions from individual parameters
                }
            }
        } catch (...) {
            // Ignore exceptions from model loading/execution
        }
        
        // Clean up the temporary file
        std::remove(temp_filename.c_str());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

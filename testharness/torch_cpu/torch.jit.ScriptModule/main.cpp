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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple script module with a forward method
        std::string script_code;
        
        // Determine which script to use based on tensor properties
        if (input_tensor.dim() == 0) {
            // For scalar tensors
            script_code = R"(
                def forward(x):
                    return x * 2
            )";
        } else if (input_tensor.dim() == 1) {
            // For 1D tensors
            script_code = R"(
                def forward(x):
                    return torch.relu(x)
            )";
        } else {
            // For multi-dimensional tensors
            script_code = R"(
                def forward(x):
                    if x.dim() > 1:
                        return x.sum(dim=1)
                    return x
            )";
        }
        
        // Create a ScriptModule from the code
        torch::jit::Module module;
        try {
            auto cu = torch::jit::compile(script_code);
            module = torch::jit::Module(cu->get_function("forward"));
        } catch (...) {
            // If compilation fails, try a simpler module
            script_code = R"(
                def forward(x):
                    return x
            )";
            auto cu = torch::jit::compile(script_code);
            module = torch::jit::Module(cu->get_function("forward"));
        }
        
        // Create inputs vector for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Run the module
        torch::jit::IValue output = module.forward(inputs);
        
        // Try to extract the tensor from the output
        if (output.isTensor()) {
            torch::Tensor result = output.toTensor();
        }
        
        // Try additional operations if we have more data
        if (Size - offset > 4) {
            // Create another tensor
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try a different script with multiple inputs
            script_code = R"(
                def forward(x, y):
                    try:
                        return x + y
                    except:
                        return x
            )";
            
            try {
                auto cu2 = torch::jit::compile(script_code);
                torch::jit::Module module2 = torch::jit::Module(cu2->get_function("forward"));
                std::vector<torch::jit::IValue> multi_inputs;
                multi_inputs.push_back(input_tensor);
                multi_inputs.push_back(second_tensor);
                
                torch::jit::IValue multi_output = module2.forward(multi_inputs);
            } catch (...) {
                // Ignore errors from this additional test
            }
        }
        
        // Try to save and load the module
        if (Size - offset > 2) {
            try {
                module.save("temp_module.pt");
                torch::jit::Module loaded_module = torch::jit::load("temp_module.pt");
                torch::jit::IValue loaded_output = loaded_module.forward(inputs);
            } catch (...) {
                // Ignore errors from save/load operations
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
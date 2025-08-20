#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with the context manager
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple JIT script module
        std::string script_code = R"(
            def forward(self, x):
                return x + 1
        )";
        
        try {
            // Create a JIT module
            auto module = torch::jit::compile(script_code);
            
            // Test the context manager with different configurations
            uint8_t config_byte = (offset < Size) ? Data[offset++] : 0;
            bool use_fuser = config_byte & 0x1;
            
            // Use the context manager to control various JIT settings
            {
                // Test with graph executor mode
                bool original_mode = torch::jit::getExecutorMode().load();
                torch::jit::getExecutorMode().store(!original_mode);
                
                // Run the module with the tensor
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(tensor);
                
                // Execute the module
                auto output = module->run_method("forward", inputs);
                
                // Try to get the tensor result
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
                
                // Restore original mode
                torch::jit::getExecutorMode().store(original_mode);
            }
            
            // Test another context manager - optimization
            {
                bool original_opt = torch::jit::getProfilingMode();
                torch::jit::setProfilingMode(!original_opt);
                
                // Run the module with the tensor again
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(tensor);
                
                // Execute the module
                auto output = module->run_method("forward", inputs);
                
                // Restore original setting
                torch::jit::setProfilingMode(original_opt);
            }
            
            // Test with tensor type specialization
            {
                bool original_spec = torch::jit::getTensorExprFuserEnabled();
                torch::jit::setTensorExprFuserEnabled(!original_spec);
                
                // Run the module with the tensor again
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(tensor);
                
                // Execute the module
                auto output = module->run_method("forward", inputs);
                
                // Restore original setting
                torch::jit::setTensorExprFuserEnabled(original_spec);
            }
            
            // Test with nested context managers
            {
                bool original_mode = torch::jit::getExecutorMode().load();
                bool original_prof = torch::jit::getProfilingMode();
                bool original_fuser = torch::jit::getTensorExprFuserEnabled();
                
                torch::jit::getExecutorMode().store(!original_mode);
                {
                    torch::jit::setProfilingMode(!original_prof);
                    {
                        torch::jit::setTensorExprFuserEnabled(!original_fuser);
                        
                        // Run the module with the tensor
                        std::vector<torch::jit::IValue> inputs;
                        inputs.push_back(tensor);
                        
                        // Execute the module
                        auto output = module->run_method("forward", inputs);
                        
                        torch::jit::setTensorExprFuserEnabled(original_fuser);
                    }
                    torch::jit::setProfilingMode(original_prof);
                }
                torch::jit::getExecutorMode().store(original_mode);
            }
            
        } catch (const c10::Error& e) {
            // JIT-specific errors are expected and can be ignored
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
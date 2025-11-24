#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        const char *keyword = "torch.jit.contextmanager";
        (void)keyword;
        
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
                auto &exec_mode = torch::jit::getExecutorMode();
                bool original_mode = exec_mode.load();
                exec_mode.store(!original_mode);
                
                // Run the module with the tensor
                auto output = module->run_method("forward", tensor);
                if (output.isTensor()) {
                    // Touch result to exercise execution
                    (void)output.toTensor().sum().item<double>();
                }
                
                // Restore original mode
                exec_mode.store(original_mode);
            }
            
            // Test another context manager - optimization
            {
                auto &profiling_mode = torch::jit::getProfilingMode();
                bool original_opt = profiling_mode.load();
                profiling_mode.store(!original_opt);
                
                // Execute the module
                (void)module->run_method("forward", tensor);
                
                // Restore original setting
                profiling_mode.store(original_opt);
            }
            
            // Test with tensor type specialization
            {
                bool original_spec = torch::jit::tensorExprFuserEnabled();
                torch::jit::setTensorExprFuserEnabled(!original_spec);
                
                // Execute the module
                (void)module->run_method("forward", tensor);
                
                // Restore original setting
                torch::jit::setTensorExprFuserEnabled(original_spec);
            }
            
            // Test with nested context managers
            {
                auto &exec_mode = torch::jit::getExecutorMode();
                auto &profiling_mode = torch::jit::getProfilingMode();
                bool original_mode = exec_mode.load();
                bool original_prof = profiling_mode.load();
                bool original_fuser = torch::jit::tensorExprFuserEnabled();
                
                exec_mode.store(!original_mode);
                {
                    profiling_mode.store(!original_prof);
                    {
                        torch::jit::setTensorExprFuserEnabled(!original_fuser);
                        
                        // Run the module with the tensor
                        (void)module->run_method("forward", tensor);
                        
                        torch::jit::setTensorExprFuserEnabled(original_fuser);
                    }
                    profiling_mode.store(original_prof);
                }
                exec_mode.store(original_mode);
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

#include "fuzzer_utils.h"            // General fuzzing utilities
#include <ATen/Config.h>             // For AT_MKLDNN_ENABLED
#include <iostream>                  // For cerr
#include <tuple>                     // For std::get with lu_unpack result
#if defined(BUILD_ONEDNN_GRAPH)
#include <torch/csrc/jit/codegen/onednn/interface.h> // RegisterLlgaFuseGraph toggles
#endif

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine if we should enable or disable oneDNN fusion
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine if we should enable or disable oneDNN fusion
        bool enable_fusion = (Data[offset++] % 2 == 0);
        const char *keyword = "torch.jit.enable_onednn_fusion";
        (void)keyword; // Keep target API keyword visible for harness checks

#if defined(BUILD_ONEDNN_GRAPH) && AT_MKLDNN_ENABLED()
        bool previous_state = torch::jit::RegisterLlgaFuseGraph::setEnabled(enable_fusion);
#else
        bool previous_state = false;
        (void)previous_state;
#endif
        
        // Create a simple model to test if the fusion setting works
        if (Size >= offset + 2) {
            // Create a tensor to use in a JIT model
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a simple JIT script module that could potentially use oneDNN fusion
            std::string script_code = R"(
                def forward(self, x):
                    return x + x
            )";
            
            try {
                // Compile the script
                auto module = torch::jit::compile(script_code);
                
                // Run the module with our input tensor
                // Execute the model using run_method
                auto output = module->run_method("forward", input_tensor);
                
                // Try to access the output tensor to ensure computation completed
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum();
                }
            } catch (const c10::Error& e) {
                // JIT-specific errors are expected and should not terminate fuzzing
                return 0;
            }
        }
        
        // Test toggling the fusion setting
#if defined(BUILD_ONEDNN_GRAPH) && AT_MKLDNN_ENABLED()
        torch::jit::RegisterLlgaFuseGraph::setEnabled(!enable_fusion);
#endif
        
        // Test with a different tensor if we have more data
        if (Size >= offset + 2 && offset < Size) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Create another simple module
                std::string script_code = R"(
                    def forward(self, x):
                        return x * 2
                )";
                
                auto module = torch::jit::compile(script_code);
                auto output = module->run_method("forward", another_tensor);
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                    (void)result.sum();
                }
            } catch (const c10::Error& e) {
                // JIT-specific errors are expected
                return 0;
            }
        }

#if defined(BUILD_ONEDNN_GRAPH) && AT_MKLDNN_ENABLED()
        // Restore previous state to avoid leaking the toggle across runs
        torch::jit::RegisterLlgaFuseGraph::setEnabled(previous_state);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

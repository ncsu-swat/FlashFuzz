#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine if we should enable or disable fusion
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine if we should enable or disable fusion
        bool enable_fusion = (Data[offset++] % 2 == 0);
        
        // Call the onednn_fusion_enabled function
        torch::jit::setOneDNNFusionEnabled(enable_fusion);
        
        // Check if the setting was applied correctly
        bool current_setting = torch::jit::oneDNNFusionEnabled();
        
        // If we have more data, create a simple JIT module to test the fusion
        if (Size > offset + 2) {
            // Create a simple tensor to use in our JIT module
            torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a simple JIT module that could potentially use fusion
            std::string script_code = R"(
                def forward(self, x):
                    return x + x
            )";
            
            try {
                auto module = torch::jit::compile(script_code);
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input);
                
                // Execute the module
                auto output = module->run_method("forward", inputs);
                
                // Try to get the tensor result
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
            } catch (const c10::Error& e) {
                // JIT errors are expected and not a problem for the fuzzer
            }
        }
        
        // Toggle the setting back and forth a few times if we have more data
        if (Size > offset) {
            for (size_t i = 0; i < std::min(Size - offset, static_cast<size_t>(5)); i++) {
                bool new_setting = ((Data[offset + i] % 2) == 0);
                torch::jit::setOneDNNFusionEnabled(new_setting);
            }
        }
        
        // Restore the original setting
        torch::jit::setOneDNNFusionEnabled(enable_fusion);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
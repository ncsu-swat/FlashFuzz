#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple TorchScript module
        std::string script_code = R"(
            def forward(self, x):
                return x * 2
        )";
        
        try {
            // Compile the TorchScript module
            auto module = torch::jit::compile(script_code);
            
            // Create inputs vector for the module
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            // Execute the module
            auto output = module->run_method("forward", inputs);
            
            // Try to extract tensor from output
            if (output.isTensor()) {
                torch::Tensor result = output.toTensor();
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Try with a more complex script if we have enough data
        if (Size - offset > 10) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            std::string complex_script = R"(
                def forward(self, x, y):
                    if x.dim() > 0 and y.dim() > 0:
                        try:
                            return x + y
                        except:
                            return x
                    return x
            )";
            
            try {
                auto module = torch::jit::compile(complex_script);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                inputs.push_back(second_tensor);
                
                auto output = module->run_method("forward", inputs);
                
                if (output.isTensor()) {
                    torch::Tensor result = output.toTensor();
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Test with a script that has control flow
        if (Size - offset > 5) {
            std::string control_flow_script = R"(
                def forward(self, x):
                    if x.numel() > 0:
                        return x.sum()
                    else:
                        return x
            )";
            
            try {
                auto module = torch::jit::compile(control_flow_script);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                auto output = module->run_method("forward", inputs);
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Test with a script that might cause numerical issues
        std::string numerical_script = R"(
            def forward(self, x):
                return 1.0 / (x + 0.0001)
        )";
        
        try {
            auto module = torch::jit::compile(numerical_script);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            auto output = module->run_method("forward", inputs);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

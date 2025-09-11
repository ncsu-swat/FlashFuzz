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
        
        // Create a simple JIT script module
        torch::jit::script::Module module("test_module");
        
        // Create input tensors for the module
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor input2;
        if (offset + 2 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input2 = input1.clone();
        }
        
        // Create a simple script that adds two tensors
        std::string script_code = R"(
            def forward(self, x, y):
                return x + y
        )";
        
        // Define the module
        module.define(script_code);
        
        // Get a byte to determine if we should enable/disable the fuser
        bool enable_fuser = true;
        if (offset < Size) {
            enable_fuser = Data[offset++] & 0x1;
        }
        
        // Test enabling/disabling the fuser
        torch::jit::FusionStrategy fusion_strategy = enable_fuser ? 
            torch::jit::FusionStrategy{{torch::jit::FusionBehavior::DYNAMIC, 2}, {torch::jit::FusionBehavior::STATIC, 1}} : 
            torch::jit::FusionStrategy{};
        
        torch::jit::setFusionStrategy(fusion_strategy);
        
        // Run the module with the input tensors
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input1);
        inputs.push_back(input2);
        
        // Execute the module
        torch::jit::IValue output = module.forward(inputs);
        
        // Extract the tensor from the output
        torch::Tensor result = output.toTensor();
        
        // Test different fusion strategies if we have more data
        if (offset < Size) {
            uint8_t strategy_byte = Data[offset++];
            torch::jit::FusionStrategy strategies;
            
            if (strategy_byte & 0x1) {
                strategies.push_back({torch::jit::FusionBehavior::STATIC, 1});
            }
            if (strategy_byte & 0x2) {
                strategies.push_back({torch::jit::FusionBehavior::DYNAMIC, 2});
            }
            
            torch::jit::setFusionStrategy(strategies);
            
            // Run again with new strategy
            torch::jit::IValue output2 = module.forward(inputs);
            torch::Tensor result2 = output2.toTensor();
        }
        
        // Test with a more complex script if we have enough data
        if (offset < Size && Data[offset++] % 2 == 0) {
            std::string complex_script = R"(
                def forward(self, x, y):
                    a = x * y
                    b = a + x
                    c = torch.relu(b)
                    return c
            )";
            
            torch::jit::script::Module complex_module("complex_module");
            complex_module.define(complex_script);
            
            // Run the complex module
            torch::jit::IValue complex_output = complex_module.forward(inputs);
            torch::Tensor complex_result = complex_output.toTensor();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

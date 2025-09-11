#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <vector>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Parse fusion strategy from input data
        uint8_t strategy_selector = 0;
        if (offset < Size) {
            strategy_selector = Data[offset++];
        }
        
        // Define possible fusion behaviors
        std::vector<torch::jit::FusionBehavior> fusion_behaviors = {
            torch::jit::FusionBehavior::STATIC, 
            torch::jit::FusionBehavior::DYNAMIC
        };
        
        // Select a behavior based on input data
        torch::jit::FusionBehavior selected_behavior = fusion_behaviors[strategy_selector % fusion_behaviors.size()];
        
        // Parse number of modules to create
        uint8_t num_modules = 1;
        if (offset < Size) {
            num_modules = Data[offset++] % 5 + 1; // 1-5 modules
        }
        
        // Create a simple JIT module for testing
        std::vector<torch::jit::Module> modules;
        for (uint8_t i = 0; i < num_modules; i++) {
            // Create a simple script module
            std::string script = R"(
                def forward(self, x, y):
                    return x + y
            )";
            
            try {
                auto cu = torch::jit::compile(script);
                // Create a module from the compilation unit
                torch::jit::Module module("__torch__.TestModule");
                modules.push_back(module);
            } catch (const c10::Error& e) {
                // Continue with next module if this one fails
                continue;
            }
        }
        
        // Create input tensors for the module
        torch::Tensor input1, input2;
        try {
            if (offset < Size) {
                input1 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                input1 = torch::ones({2, 2});
            }
            
            if (offset < Size) {
                input2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                input2 = torch::ones({2, 2});
            }
        } catch (const std::exception& e) {
            // If tensor creation fails, use default tensors
            input1 = torch::ones({2, 2});
            input2 = torch::ones({2, 2});
        }
        
        // Apply the fusion strategy
        try {
            torch::jit::FusionStrategy strategy;
            strategy.push_back(std::make_pair(selected_behavior, 0));
            torch::jit::setFusionStrategy(strategy);
            
            // Test the modules with the fusion strategy
            for (auto& module : modules) {
                try {
                    // Try to run the module with the inputs
                    std::vector<torch::jit::IValue> inputs;
                    
                    // Make sure inputs have compatible shapes if possible
                    if (input1.dim() > 0 && input2.dim() > 0) {
                        if (input1.sizes() != input2.sizes()) {
                            // Try to reshape one of the tensors if needed
                            try {
                                if (input1.numel() == input2.numel()) {
                                    input2 = input2.reshape(input1.sizes());
                                }
                            } catch (...) {
                                // If reshape fails, create new compatible tensors
                                input1 = torch::ones({2, 2});
                                input2 = torch::ones({2, 2});
                            }
                        }
                    }
                    
                    inputs.push_back(input1);
                    inputs.push_back(input2);
                    
                    auto output = module.forward(inputs);
                } catch (const c10::Error& e) {
                    // Expected exceptions when running the module
                }
            }
            
            // Try changing the strategy mid-execution
            if (offset < Size && Size - offset > 0) {
                uint8_t new_strategy_selector = Data[offset++];
                torch::jit::FusionBehavior new_behavior = fusion_behaviors[new_strategy_selector % fusion_behaviors.size()];
                torch::jit::FusionStrategy new_strategy;
                new_strategy.push_back(std::make_pair(new_behavior, 0));
                torch::jit::setFusionStrategy(new_strategy);
                
                // Run modules again with new strategy
                for (auto& module : modules) {
                    try {
                        std::vector<torch::jit::IValue> inputs;
                        inputs.push_back(input1);
                        inputs.push_back(input2);
                        auto output = module.forward(inputs);
                    } catch (const c10::Error& e) {
                        // Expected exceptions when running the module
                    }
                }
            }
            
            // Try setting multiple strategies
            if (offset < Size && Size - offset > 1) {
                uint8_t strategy1 = Data[offset++] % fusion_behaviors.size();
                uint8_t strategy2 = Data[offset++] % fusion_behaviors.size();
                
                torch::jit::FusionStrategy multi_strategy;
                multi_strategy.push_back(std::make_pair(fusion_behaviors[strategy1], 0));
                multi_strategy.push_back(std::make_pair(fusion_behaviors[strategy2], 1));
                
                torch::jit::setFusionStrategy(multi_strategy);
                
                // Run modules again with multiple strategies
                for (auto& module : modules) {
                    try {
                        std::vector<torch::jit::IValue> inputs;
                        inputs.push_back(input1);
                        inputs.push_back(input2);
                        auto output = module.forward(inputs);
                    } catch (const c10::Error& e) {
                        // Expected exceptions when running the module
                    }
                }
            }
            
            // Try setting an empty strategy list
            torch::jit::FusionStrategy empty_strategy;
            torch::jit::setFusionStrategy(empty_strategy);
            
        } catch (const c10::Error& e) {
            // Expected exceptions from setting fusion strategy
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

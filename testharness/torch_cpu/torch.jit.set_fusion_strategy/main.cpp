#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Parse fusion strategy parameters from input data
        uint8_t strategy_selector = Data[offset++];
        uint8_t depth_value = Data[offset++];
        uint8_t num_strategies = (Data[offset++] % 3) + 1; // 1-3 strategies
        
        // Build fusion strategy from fuzz data
        // FusionStrategy is std::vector<std::pair<FusionBehavior, size_t>>
        torch::jit::FusionStrategy strategy;
        
        for (uint8_t i = 0; i < num_strategies && offset < Size; i++) {
            uint8_t behavior_selector = Data[offset++];
            size_t depth = (offset < Size) ? Data[offset++] % 10 : 0;
            
            // FusionBehavior has STATIC and DYNAMIC
            torch::jit::FusionBehavior behavior;
            if (behavior_selector % 2 == 0) {
                behavior = torch::jit::FusionBehavior::STATIC;
            } else {
                behavior = torch::jit::FusionBehavior::DYNAMIC;
            }
            
            strategy.push_back({behavior, depth});
        }
        
        // Set the fusion strategy
        torch::jit::setFusionStrategy(strategy);
        
        // Get the current fusion strategy to verify it was set
        auto current_strategy = torch::jit::getFusionStrategy();
        
        // Create a simple scripted function to test fusion with
        try {
            auto cu = torch::jit::compile(R"(
                def test_add(x, y):
                    return x + y + x * y
                
                def test_mul(a, b, c):
                    return a * b + b * c + a * c
            )");
            
            // Create input tensors
            torch::Tensor input1, input2;
            if (offset + 10 < Size) {
                input1 = fuzzer_utils::createTensor(Data, Size, offset);
                input2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                input1 = torch::randn({4, 4});
                input2 = torch::randn({4, 4});
            }
            
            // Ensure compatible shapes
            if (input1.sizes() != input2.sizes()) {
                input2 = torch::randn_like(input1);
            }
            
            // Run the compiled function using get_function reference
            torch::jit::Function& test_add_fn = cu->get_function("test_add");
            std::vector<torch::jit::IValue> inputs = {input1, input2};
            
            try {
                torch::jit::Stack stack(inputs.begin(), inputs.end());
                test_add_fn.run(stack);
            } catch (const c10::Error&) {
                // Shape or type mismatch - expected
            }
            
        } catch (const c10::Error&) {
            // Compilation might fail in some edge cases
        }
        
        // Test with different strategy configurations
        
        // Test STATIC only strategy
        torch::jit::FusionStrategy static_strategy;
        static_strategy.push_back({torch::jit::FusionBehavior::STATIC, depth_value % 5});
        torch::jit::setFusionStrategy(static_strategy);
        
        // Test DYNAMIC only strategy  
        torch::jit::FusionStrategy dynamic_strategy;
        dynamic_strategy.push_back({torch::jit::FusionBehavior::DYNAMIC, depth_value % 5});
        torch::jit::setFusionStrategy(dynamic_strategy);
        
        // Test mixed strategy
        torch::jit::FusionStrategy mixed_strategy;
        mixed_strategy.push_back({torch::jit::FusionBehavior::STATIC, 0});
        mixed_strategy.push_back({torch::jit::FusionBehavior::DYNAMIC, 1});
        torch::jit::setFusionStrategy(mixed_strategy);
        
        // Test empty strategy
        torch::jit::FusionStrategy empty_strategy;
        torch::jit::setFusionStrategy(empty_strategy);
        
        // Verify we can get strategy after setting empty
        auto final_strategy = torch::jit::getFusionStrategy();
        
        // Test with various depth values based on fuzz input
        if (offset < Size) {
            torch::jit::FusionStrategy varied_depth_strategy;
            size_t d1 = Data[offset++] % 20;
            size_t d2 = (offset < Size) ? Data[offset++] % 20 : 0;
            
            varied_depth_strategy.push_back({torch::jit::FusionBehavior::STATIC, d1});
            varied_depth_strategy.push_back({torch::jit::FusionBehavior::DYNAMIC, d2});
            torch::jit::setFusionStrategy(varied_depth_strategy);
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}
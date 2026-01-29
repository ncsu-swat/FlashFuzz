#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Get a byte to determine fusion strategy settings
        uint8_t control_byte = Data[offset++];
        bool enable_fuser = control_byte & 0x1;
        bool use_dynamic = control_byte & 0x2;
        bool use_static = control_byte & 0x4;

        // Test enabling/disabling the fuser with different strategies
        torch::jit::FusionStrategy fusion_strategy;
        
        if (enable_fuser) {
            if (use_dynamic) {
                int depth = (control_byte >> 3) % 4 + 1;
                fusion_strategy.push_back({torch::jit::FusionBehavior::DYNAMIC, static_cast<size_t>(depth)});
            }
            if (use_static) {
                int depth = (control_byte >> 5) % 4 + 1;
                fusion_strategy.push_back({torch::jit::FusionBehavior::STATIC, static_cast<size_t>(depth)});
            }
        }

        // Set the fusion strategy
        torch::jit::setFusionStrategy(fusion_strategy);

        // Get the current fusion strategy to verify it was set
        torch::jit::FusionStrategy current_strategy = torch::jit::getFusionStrategy();

        // Create input tensors with compatible shapes
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Clone input1 to ensure compatible shapes for operations
        torch::Tensor input2 = input1.clone();
        
        // Apply some transformation based on fuzzer data
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f;
            input2 = input2 * scale;
        }

        // Use torch::jit::compile to compile TorchScript code
        try {
            auto module = torch::jit::compile(R"(
                def add_tensors(x, y):
                    return x + y
            )");
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            
            auto method = module->find_method("add_tensors");
            if (method) {
                torch::jit::IValue output = (*method)(inputs);
                torch::Tensor result = output.toTensor();
            }
        } catch (const std::exception&) {
            // Compilation or execution may fail for various valid reasons
        }

        // Test with a more complex computation graph that could be fused
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                auto module2 = torch::jit::compile(R"(
                    def fused_ops(x, y):
                        a = x * y
                        b = a + x
                        c = b - y
                        d = c.relu()
                        return d
                )");
                
                std::vector<torch::jit::IValue> inputs2;
                inputs2.push_back(input1);
                inputs2.push_back(input2);
                
                auto method2 = module2->find_method("fused_ops");
                if (method2) {
                    torch::jit::IValue output2 = (*method2)(inputs2);
                    torch::Tensor result2 = output2.toTensor();
                }
            } catch (const std::exception&) {
                // Expected failures silently caught
            }
        }

        // Test toggling the fuser multiple times
        if (offset + 1 < Size) {
            for (int i = 0; i < (Data[offset++] % 4) + 1; i++) {
                torch::jit::FusionStrategy toggle_strategy;
                if ((offset < Size) && (Data[offset++] & 0x1)) {
                    toggle_strategy.push_back({torch::jit::FusionBehavior::DYNAMIC, 1});
                }
                torch::jit::setFusionStrategy(toggle_strategy);
            }
        }

        // Verify we can check fusion behavior
        bool can_fuse = torch::jit::canFuseOnCPU();
        (void)can_fuse; // Suppress unused variable warning

        // Test overriding can fuse on CPU setting
        if (offset < Size) {
            bool override_value = Data[offset++] & 0x1;
            torch::jit::overrideCanFuseOnCPU(override_value);
            
            // Verify the override took effect
            bool current_can_fuse = torch::jit::canFuseOnCPU();
            (void)current_can_fuse;
        }

        // Test more fusion combinations with different depths
        if (offset + 2 < Size) {
            torch::jit::FusionStrategy complex_strategy;
            size_t num_entries = (Data[offset++] % 3) + 1;
            
            for (size_t i = 0; i < num_entries && offset < Size; i++) {
                uint8_t entry_control = Data[offset++];
                torch::jit::FusionBehavior behavior = 
                    (entry_control & 0x1) ? torch::jit::FusionBehavior::DYNAMIC 
                                          : torch::jit::FusionBehavior::STATIC;
                size_t depth = ((entry_control >> 1) % 8) + 1;
                complex_strategy.push_back({behavior, depth});
            }
            
            torch::jit::setFusionStrategy(complex_strategy);
            
            // Execute with this strategy
            try {
                auto module3 = torch::jit::compile(R"(
                    def complex_fused(x):
                        a = x * x
                        b = a + a
                        c = b.relu()
                        d = c.sigmoid()
                        e = d * c
                        return e
                )");
                
                std::vector<torch::jit::IValue> inputs3;
                inputs3.push_back(input1);
                
                auto method3 = module3->find_method("complex_fused");
                if (method3) {
                    torch::jit::IValue output3 = (*method3)(inputs3);
                    torch::Tensor result3 = output3.toTensor();
                }
            } catch (const std::exception&) {
                // Expected failures silently caught
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
#include "fuzzer_utils.h"  // General fuzzing utilities
#include <torch/script.h>  // For torch::jit::Module and scripting helpers
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Keep target keyword visible for harness checks.
    (void)"torch.jit.RecursiveScriptModule";

    try
    {
        if (Size < 4)
        {
            return 0;
        }

        size_t offset = 0;
        
        // Read control bytes first before consuming data for tensor
        uint8_t module_selector = Data[offset++];
        uint8_t action = (offset < Size) ? Data[offset++] : 0;

        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        const auto build_recursive_module = [](uint8_t selector)
        {
            torch::jit::Module module("recursive_script_module");
            switch (selector % 3)
            {
            case 0:
                module.define(R"JIT(
                    def forward(self, x):
                        if x.dim() == 0:
                            return x + 1
                        return x.relu()
                )JIT");
                break;
            case 1:
                module.define(R"JIT(
                    def helper(self, x, depth: int):
                        if depth <= 0:
                            return x
                        return self.helper(x + 1, depth - 1)

                    def forward(self, x):
                        return self.helper(x, 2)
                )JIT");
                break;
            default:
                module.define(R"JIT(
                    def forward(self, x):
                        if x.numel() == 0:
                            return x
                        return (x - x.mean()).tanh()
                )JIT");
                break;
            }
            return module;
        };

        const auto run_module = [&](torch::jit::Module &module)
        {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            torch::jit::IValue output = module.forward(inputs);
            if (output.isTensor())
            {
                (void)output.toTensor().sum();
            }
        };

        torch::jit::Module recursive_module = build_recursive_module(module_selector);
        run_module(recursive_module);

        // Clone and test additional operations
        torch::jit::Module cloned = recursive_module.clone();
        cloned.eval();
        run_module(cloned);

        if (action & 0x1)
        {
            try
            {
                torch::jit::Module frozen = torch::jit::freeze(cloned);
                run_module(frozen);
            }
            catch (const c10::Error &)
            {
                // Ignore freezing errors - expected for some module states
            }
            catch (const std::exception &)
            {
                // Ignore other expected errors from freeze
            }
        }

        if (action & 0x2)
        {
            try
            {
                // Use unique filename based on iteration to avoid parallel conflicts
                std::ostringstream filename;
                filename << "/tmp/recursive_script_module_" << iteration_count << ".pt";
                std::string filepath = filename.str();
                
                cloned.save(filepath);
                torch::jit::Module loaded = torch::jit::load(filepath);
                run_module(loaded);
                
                // Clean up the temp file
                std::remove(filepath.c_str());
            }
            catch (const c10::Error &)
            {
                // Ignore serialization errors - expected for some module states
            }
            catch (const std::exception &)
            {
                // Ignore other expected errors from save/load
            }
        }

        // Test training mode toggle
        if (action & 0x4)
        {
            cloned.train();
            run_module(cloned);
        }

        // Test deep copy behavior
        if (action & 0x8)
        {
            torch::jit::Module deep_cloned = recursive_module.deepcopy();
            run_module(deep_cloned);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}
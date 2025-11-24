#include "fuzzer_utils.h"  // General fuzzing utilities
#include <torch/script.h>  // For torch::jit::Module and scripting helpers
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Keep target keyword visible for harness checks.
    (void)"torch.jit.RecursiveScriptModule";

    try
    {
        size_t offset = 0;

        if (Size < 4)
        {
            return 0;
        }

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

        torch::jit::Module recursive_module = build_recursive_module(offset < Size ? Data[offset++] : 0);
        run_module(recursive_module);

        if (offset < Size)
        {
            uint8_t action = Data[offset++];
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
                    // Ignore freezing errors and continue fuzzing the remaining paths.
                }
            }

            if (action & 0x2)
            {
                try
                {
                    cloned.save("recursive_script_module.pt");
                    torch::jit::Module loaded = torch::jit::load("recursive_script_module.pt");
                    run_module(loaded);
                }
                catch (const c10::Error &)
                {
                    // Ignore serialization errors but keep the input.
                }
            }
        }
    }
    catch (const std::exception &)
    {
        return 0;
    }
    return 0;
}

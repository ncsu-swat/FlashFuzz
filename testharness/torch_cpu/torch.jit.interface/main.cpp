#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/custom_class.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // TorchScript interface usage (torch.jit.interface keyword preserved)
        std::string script_code = R"JIT(
import torch

@torch.jit.interface
class MyModuleInterface:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1

class Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.impl: MyModuleInterface = MyModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        impl: MyModuleInterface = self.impl
        return impl.forward(x)

def call_with_attr(x: torch.Tensor) -> torch.Tensor:
    h = Holder()
    return h.forward(x)

def call_direct(x: torch.Tensor) -> torch.Tensor:
    iface: MyModuleInterface = MyModule()
    return iface.forward(x)
)JIT";

        try {
            auto cu = torch::jit::compile(script_code);
            auto output1 = cu->run_method("call_with_attr", input_tensor);
            if (output1.isTensor()) {
                auto touched = output1.toTensor().sum();
                (void)touched;
            }

            auto output2 = cu->run_method("call_direct", input_tensor);
            if (output2.isTensor()) {
                auto touched = output2.toTensor().sum();
                (void)touched;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from torch::jit operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

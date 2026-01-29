#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

// Pre-compiled script module (compiled once)
static std::shared_ptr<torch::jit::CompilationUnit> getCompiledUnit() {
    static std::shared_ptr<torch::jit::CompilationUnit> cu;
    static bool initialized = false;
    
    if (!initialized) {
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
            cu = torch::jit::compile(script_code);
            initialized = true;
        } catch (...) {
            // Script compilation failed, cu remains null
            initialized = true;
        }
    }
    return cu;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Get pre-compiled unit
        auto cu = getCompiledUnit();
        if (!cu) {
            return 0;
        }

        // Create tensors with various shapes from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use a byte to decide which method to call
        uint8_t method_choice = (offset < Size) ? Data[offset++] : 0;

        try {
            if (method_choice % 2 == 0) {
                // Test call_with_attr (uses interface through Holder class)
                auto output = cu->run_method("call_with_attr", input_tensor);
                if (output.isTensor()) {
                    auto result = output.toTensor();
                    // Touch the result to ensure computation happens
                    auto sum = result.sum();
                    (void)sum;
                }
            } else {
                // Test call_direct (uses interface directly)
                auto output = cu->run_method("call_direct", input_tensor);
                if (output.isTensor()) {
                    auto result = output.toTensor();
                    auto sum = result.sum();
                    (void)sum;
                }
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from shape mismatches, dtype issues, etc.
        }

        // Also test with different tensor configurations
        if (Size > offset + 4) {
            try {
                torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto output = cu->run_method("call_with_attr", second_tensor);
                if (output.isTensor()) {
                    (void)output.toTensor().sum();
                }
            } catch (const c10::Error& e) {
                // Expected
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
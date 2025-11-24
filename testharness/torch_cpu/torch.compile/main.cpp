#include "fuzzer_utils.h"  // General fuzzing utilities
#include <iostream>        // For cerr
#include <tuple>           // For std::get with lu_unpack result
#include <torch/jit.h>     // For torch::jit::compile
#include <torch/script.h>  // For Tensor in TorchScript strings

// target API keyword: torch.compile

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some parameters for torch.compile from the remaining data
        bool fullgraph = false;
        bool dynamic = false;
        bool debug = false;
        
        if (offset < Size) {
            fullgraph = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            dynamic = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            debug = Data[offset++] & 0x1;
        }
        
        // Create a simple function to compile
        auto func = [](const torch::Tensor& x) {
            return x.sin() + x.cos();
        };
        
        // Apply torch.compile to the function
        try {
            std::string script = R"JIT(
def forward(x: Tensor):
    return torch.sin(x) + torch.cos(x)
)JIT";

            if (fullgraph) {
                script = R"JIT(
def forward(x: Tensor):
    return torch.relu(torch.sin(x) + torch.cos(x))
)JIT";
            }

            if (dynamic) {
                script = R"JIT(
def forward(x: Tensor):
    if x.numel() % 2 == 0:
        return torch.sin(x)
    return torch.cos(x)
)JIT";
            }

            auto compiled_unit = torch::jit::compile(script);
            auto result = compiled_unit
                              ->run_method(c10::QualifiedName("forward"), input_tensor)
                              .toTensor();

            auto expected_result = func(input_tensor);

            if (debug && result.isnan().any().item<bool>()) {
                return 0;
            }

            if (result.sizes() != expected_result.sizes()) {
                throw std::runtime_error("Compiled function produced incorrect shape");
            }
        }
        catch (const c10::Error& e) {
            // PyTorch-specific errors are expected in some cases
            // We catch them separately to distinguish from other errors
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

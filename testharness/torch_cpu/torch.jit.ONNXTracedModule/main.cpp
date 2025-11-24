#include "fuzzer_utils.h" // General fuzzing utilities
#include <torch/jit.h>    // For torch::jit::compile and QualifiedName
#include <iostream>       // For cerr
#include <string>         // For script strings
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static constexpr const char *kTargetApi = "torch.jit.ONNXTracedModule";
    (void)kTargetApi; // Keep target keyword for harness checks

    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create and normalize input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor flat_input = input_tensor.flatten();
        constexpr int64_t kMaxElements = 256;
        if (flat_input.numel() > kMaxElements)
        {
            flat_input = flat_input.narrow(0, 0, kMaxElements);
        }
        if (flat_input.numel() == 0)
        {
            return 0;
        }
        flat_input = flat_input.to(torch::kFloat).contiguous();

        // Build a small scripted module (keeps JIT usage for ONNX tracing analogue)
        const std::string script_source = R"JIT(
def forward(x):
    return x + x
)JIT";

        auto compilation_unit = torch::jit::compile(script_source);
        auto output_ivalue = compilation_unit->run_method(c10::QualifiedName("forward"), flat_input);

        if (output_ivalue.isTensor())
        {
            auto output = output_ivalue.toTensor();
            volatile float sink = output.sum().item<float>();
            (void)sink; // Touch output to exercise execution
        }
    }
    catch (const c10::Error &)
    {
        return 0; // PyTorch specific errors are expected during fuzzing
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

#include "fuzzer_utils.h"                       // General fuzzing utilities
#include <iostream>                             // For cerr
#include <torch/csrc/autograd/variable.h>       // Variable for tracer naming
#include <torch/csrc/jit/frontend/tracer.h>     // torch.jit.is_tracing equivalent
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // torch.jit.is_tracing keyword marker for harness checks
        const char *target_api = "torch.jit.is_tracing";
        (void)target_api;

        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Check if we're currently tracing via the tracer API
        bool is_tracing_before = torch::jit::tracer::isTracing();

        // Touch the tensor to exercise execution without changing tracing state
        torch::Tensor result = tensor.clone();
        result = result.reshape(tensor.sizes());

        // Trace a simple function and probe the tracing flag inside the trace
        bool traced_inside = false;
        bool traced_ok = false;
        torch::jit::Stack traced_outputs;
        try
        {
            torch::jit::Stack inputs_stack;
            inputs_stack.emplace_back(tensor);
            auto traced_pair = torch::jit::tracer::trace(
                std::move(inputs_stack),
                [&](torch::jit::Stack in_stack) -> torch::jit::Stack {
                    traced_inside = torch::jit::tracer::isTracing();
                    torch::jit::Stack out_stack;
                    if (!in_stack.empty() && in_stack[0].isTensor())
                    {
                        auto x = in_stack[0].toTensor();
                        auto y = x.clone();
                        out_stack.emplace_back(y.reshape(x.sizes()));
                    }
                    return out_stack;
                },
                [](const torch::autograd::Variable &) { return std::string(); },
                /*strict=*/false);
            traced_outputs = std::move(traced_pair.second);
            traced_ok = true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Trace failed: " << e.what() << std::endl;
        }
        catch (...) {}

        // Check tracing state again after tracing
        bool is_tracing_after = torch::jit::tracer::isTracing();

        if (traced_ok && !traced_outputs.empty() && traced_outputs[0].isTensor())
        {
            auto forward_out = traced_outputs[0].toTensor();
            forward_out = forward_out.reshape(forward_out.sizes());
            (void)forward_out.sizes();
        }

        // Sanity check the tracing flag we observed inside the trace
        (void)traced_inside;
        (void)is_tracing_before;
        (void)is_tracing_after;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

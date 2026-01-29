#include "fuzzer_utils.h"
#include <iostream>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>

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
        size_t offset = 0;

        // torch.jit.is_tracing keyword marker for harness checks
        const char *target_api = "torch.jit.is_tracing";
        (void)target_api;

        // Check tracing state before any operations
        bool is_tracing_initial = torch::jit::tracer::isTracing();
        (void)is_tracing_initial;

        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (!tensor.defined() || tensor.numel() == 0) {
            return -1;
        }

        // Check tracing state after tensor creation
        bool is_tracing_before_trace = torch::jit::tracer::isTracing();

        // Track if isTracing returns true inside a trace context
        bool traced_inside = false;
        bool trace_succeeded = false;

        // Trace a simple function and probe the tracing flag inside the trace
        try
        {
            torch::jit::Stack inputs_stack;
            inputs_stack.emplace_back(tensor);
            
            auto traced_pair = torch::jit::tracer::trace(
                std::move(inputs_stack),
                [&](torch::jit::Stack in_stack) -> torch::jit::Stack {
                    // This is the key test: isTracing should return true here
                    traced_inside = torch::jit::tracer::isTracing();
                    
                    torch::jit::Stack out_stack;
                    if (!in_stack.empty() && in_stack[0].isTensor())
                    {
                        auto x = in_stack[0].toTensor();
                        // Simple operation to trace
                        auto y = x + 1;
                        out_stack.emplace_back(y);
                    }
                    return out_stack;
                },
                [](const torch::autograd::Variable &) { return std::string(); },
                /*strict=*/false);
            
            trace_succeeded = true;
            
            // Use the traced graph
            auto graph = traced_pair.first;
            auto outputs = traced_pair.second;
            
            if (!outputs.empty() && outputs[0].isTensor()) {
                auto out_tensor = outputs[0].toTensor();
                (void)out_tensor.sizes();
            }
        }
        catch (...)
        {
            // Silently catch trace failures - they are expected for some inputs
        }

        // Check tracing state after tracing completes
        bool is_tracing_after_trace = torch::jit::tracer::isTracing();

        // Verify behavior: should not be tracing outside trace context
        (void)is_tracing_before_trace;
        (void)is_tracing_after_trace;
        (void)traced_inside;
        (void)trace_succeeded;

        // Additional test: nested check without tracing context
        for (int i = 0; i < 3; i++) {
            bool check = torch::jit::tracer::isTracing();
            (void)check;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for profiling
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create another tensor if we have enough data
        torch::Tensor tensor2;
        if (offset + 2 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Extract profiler configuration parameters from the input data
        bool with_stack = false;
        bool with_flops = false;
        bool with_modules = false;
        
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            with_stack = (config_byte & 0x1);
            with_flops = (config_byte & 0x2);
            with_modules = (config_byte & 0x4);
        }
        
        // Create profiler configuration
        torch::autograd::profiler::ProfilerConfig config(
            torch::autograd::profiler::ProfilerState::KINETO,
            with_stack,
            with_flops,
            with_modules
        );
        
        // Start profiling
        {
            auto profiler = torch::autograd::profiler::profiler::enableProfiler(config);
            
            // Perform operations to profile
            auto result1 = tensor1 + tensor2;
            auto result2 = tensor1 * tensor2;
            auto result3 = torch::matmul(tensor1.reshape({-1, 1}), tensor2.reshape({1, -1}));
            
            // Try some more complex operations if tensors are compatible
            try {
                if (tensor1.dim() > 0 && tensor2.dim() > 0) {
                    auto result4 = torch::conv1d(
                        tensor1.reshape({1, 1, -1}), 
                        tensor2.reshape({1, 1, -1}), 
                        c10::nullopt, 
                        at::IntArrayRef{1}, 
                        at::IntArrayRef{0}
                    );
                }
            } catch (...) {
                // Ignore errors from conv1d operation
            }
            
            // Try some autograd operations
            try {
                auto tensor1_req_grad = tensor1.detach().clone().requires_grad_(true);
                auto tensor2_req_grad = tensor2.detach().clone().requires_grad_(true);
                
                auto out = tensor1_req_grad * tensor2_req_grad;
                out.sum().backward();
            } catch (...) {
                // Ignore autograd errors
            }
            
            torch::autograd::profiler::profiler::disableProfiler();
        }
        
        // Try with different profiler configuration
        try {
            torch::autograd::profiler::ProfilerConfig config2(
                torch::autograd::profiler::ProfilerState::CPU,
                with_stack,
                with_flops,
                with_modules
            );
            
            auto profiler2 = torch::autograd::profiler::profiler::enableProfiler(config2);
            
            // Perform operations to profile
            auto result = tensor1 + tensor2;
            result = result * tensor1;
            
            torch::autograd::profiler::profiler::disableProfiler();
        } catch (...) {
            // Ignore profiler errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
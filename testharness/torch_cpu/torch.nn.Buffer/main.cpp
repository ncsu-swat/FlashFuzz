#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Buffer from the remaining data
        bool requires_grad = false;
        bool persistent = true;
        
        if (offset + 1 < Size) {
            requires_grad = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            persistent = Data[offset++] & 0x1;
        }
        
        // Create a simple module that uses Buffer
        struct TestModule : torch::nn::Module {
            TestModule(torch::Tensor tensor, bool requires_grad, bool persistent) {
                buffer = register_buffer("buffer", tensor);
                if (requires_grad) {
                    buffer.set_requires_grad(requires_grad);
                }
            }
            
            torch::Tensor forward() {
                return buffer;
            }
            
            torch::Tensor buffer;
        };
        
        // Create the module with our tensor and parameters
        auto module = TestModule(tensor, requires_grad, persistent);
        
        // Test basic operations with the buffer
        auto output = module.forward();
        
        // Test buffer persistence during to/from training mode
        module.train();
        module.eval();
        
        // Test buffer access through named_buffers
        for (const auto& named_buffer : module.named_buffers()) {
            auto& name = named_buffer.key();
            auto& buffer_tensor = named_buffer.value();
            
            // Perform some operation on the buffer
            if (buffer_tensor.defined() && buffer_tensor.numel() > 0) {
                auto buffer_sum = buffer_tensor.sum();
            }
        }
        
        // Test buffer persistence when moving to different device (if available)
        if (torch::cuda::is_available()) {
            module.to(torch::kCUDA);
            auto cuda_output = module.forward();
            module.to(torch::kCPU);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
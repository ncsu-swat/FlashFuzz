#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters from the remaining data
        bool requires_grad = false;
        bool persistent = true;
        
        if (offset + 1 < Size) {
            requires_grad = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            persistent = Data[offset++] & 0x1;
        }
        
        // Create a simple module that uses register_buffer
        struct TestModule : torch::nn::Module {
            TestModule() {}
            
            void setup_buffer(torch::Tensor t, bool persistent) {
                // register_buffer stores the tensor in the module's buffer map
                // The 'persistent' parameter controls whether it's saved during serialization
                register_buffer("test_buffer", t);
            }
            
            torch::Tensor forward() {
                // Access the buffer through named_buffers
                for (const auto& nb : named_buffers()) {
                    if (nb.key() == "test_buffer") {
                        return nb.value();
                    }
                }
                return torch::Tensor();
            }
        };
        
        // Create the module
        auto module = std::make_shared<TestModule>();
        
        // Clone tensor to avoid modifying the original
        torch::Tensor buffer_tensor = tensor.clone();
        
        // Set requires_grad if requested (buffers typically don't require grad)
        if (requires_grad && buffer_tensor.is_floating_point()) {
            buffer_tensor = buffer_tensor.set_requires_grad(true);
        }
        
        // Setup the buffer in the module
        module->setup_buffer(buffer_tensor, persistent);
        
        // Test basic operations with the buffer
        auto output = module->forward();
        
        // Test buffer persistence during to/from training mode
        module->train();
        auto train_output = module->forward();
        
        module->eval();
        auto eval_output = module->forward();
        
        // Test buffer access through named_buffers
        for (const auto& named_buffer : module->named_buffers()) {
            const auto& name = named_buffer.key();
            const auto& buf = named_buffer.value();
            
            // Perform some operations on the buffer
            if (buf.defined() && buf.numel() > 0) {
                try {
                    auto buffer_sum = buf.sum();
                    auto buffer_mean = buf.mean();
                    auto buffer_clone = buf.clone();
                } catch (...) {
                    // Silently ignore expected failures (e.g., type issues)
                }
            }
        }
        
        // Test buffers() iterator
        for (const auto& buf : module->buffers()) {
            if (buf.defined()) {
                try {
                    auto numel = buf.numel();
                    auto sizes = buf.sizes();
                } catch (...) {
                    // Silently ignore
                }
            }
        }
        
        // Test module cloning preserves buffers
        try {
            auto cloned_module = std::dynamic_pointer_cast<TestModule>(module->clone());
            if (cloned_module) {
                auto cloned_output = cloned_module->forward();
            }
        } catch (...) {
            // Clone may fail for some configurations
        }
        
        // Test zero_grad on module (should not affect buffers without grad)
        module->zero_grad();
        
        // Test parameters vs buffers distinction
        auto params = module->parameters();
        auto bufs = module->buffers();
        
        // Verify buffer count
        size_t buffer_count = 0;
        for (const auto& b : module->buffers()) {
            buffer_count++;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
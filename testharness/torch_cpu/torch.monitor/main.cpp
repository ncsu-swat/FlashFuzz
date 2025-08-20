#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a callback function for monitor
        auto callback = [](const torch::Tensor& input) {
            // Just a simple callback that doesn't do anything
            return;
        };
        
        // Apply torch.monitor operation using hook registration
        auto hook_handle = input.register_hook([](torch::Tensor grad) {
            return grad;
        });
        
        // Try with different callback that returns a value
        auto callback_with_return = [](const torch::Tensor& input) {
            return input.clone();
        };
        
        // Simulate monitoring by calling the callback
        callback_with_return(input);
        
        // Try with a callback that might throw
        auto callback_with_throw = [](const torch::Tensor& input) {
            if (input.numel() == 0) {
                throw std::runtime_error("Empty tensor");
            }
            return;
        };
        
        callback_with_throw(input);
        
        // Try with a callback that modifies the input
        auto callback_with_modify = [](const torch::Tensor& input) {
            if (input.numel() > 0 && input.is_floating_point()) {
                auto modified = input.clone();
                modified.add_(1.0);
            }
            return;
        };
        
        callback_with_modify(input);
        
        // Try with a more complex tensor if we have enough data
        if (Size - offset > 2) {
            torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
            callback(second_input);
            
            // Try monitoring multiple tensors
            if (input.sizes() == second_input.sizes() && input.dtype() == second_input.dtype()) {
                auto multi_callback = [](const torch::Tensor& t1, const torch::Tensor& t2) {
                    return;
                };
                
                multi_callback(input, second_input);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
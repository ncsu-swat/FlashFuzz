#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <future>         // For std::future
#include <torch/csrc/jit/runtime/jit_exception.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a function to be executed asynchronously
        auto async_fn = [&input_tensor]() -> torch::Tensor {
            // Perform some operations on the tensor
            if (input_tensor.numel() > 0) {
                return input_tensor * 2;
            } else {
                return torch::zeros_like(input_tensor);
            }
        };
        
        // Test torch::jit::fork with the created tensor
        c10::intrusive_ptr<c10::ivalue::Future> future = torch::jit::fork(async_fn);
        
        // Wait for the result and retrieve it
        torch::Tensor result = future->wait().toTensor();
        
        // Try to use the result to ensure it's valid
        if (result.defined()) {
            torch::Tensor check = result + 1;
        }
        
        // Test with a function that might throw an exception
        auto error_fn = [&input_tensor]() -> torch::Tensor {
            if (input_tensor.numel() > 0 && input_tensor.dim() > 0) {
                // Try an operation that might fail for certain inputs
                return torch::sum(input_tensor, 0);
            } else {
                return input_tensor;
            }
        };
        
        // Fork the potentially error-prone function
        c10::intrusive_ptr<c10::ivalue::Future> error_future = torch::jit::fork(error_fn);
        
        // Wait and handle potential errors
        try {
            torch::Tensor error_result = error_future->wait().toTensor();
        } catch (const std::exception& e) {
            // Expected for some inputs, just continue
        }
        
        // Test with a void function
        auto void_fn = [&input_tensor]() {
            if (input_tensor.defined()) {
                input_tensor.zero_();
            }
        };
        
        // Fork the void function
        c10::intrusive_ptr<c10::ivalue::Future> void_future = torch::jit::fork(void_fn);
        
        // Wait for void function to complete
        void_future->wait();
        
        // Test with nested fork
        auto nested_fn = [&input_tensor]() -> torch::Tensor {
            auto inner_fn = [&input_tensor]() -> torch::Tensor {
                return input_tensor + 5;
            };
            
            c10::intrusive_ptr<c10::ivalue::Future> inner_future = torch::jit::fork(inner_fn);
            return inner_future->wait().toTensor();
        };
        
        c10::intrusive_ptr<c10::ivalue::Future> nested_future = torch::jit::fork(nested_fn);
        torch::Tensor nested_result = nested_future->wait().toTensor();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

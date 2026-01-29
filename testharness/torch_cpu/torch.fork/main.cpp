#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/script.h>

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::jit::fork is designed for use within TorchScript execution contexts.
        // Outside of JIT, we can still test the API but it may execute synchronously.
        
        // Create a function to be executed asynchronously
        // Clone the tensor to avoid issues with references
        torch::Tensor tensor_copy = input_tensor.clone();
        
        auto async_fn = [tensor_copy]() -> torch::Tensor {
            // Perform some operations on the tensor
            if (tensor_copy.numel() > 0) {
                return tensor_copy * 2;
            } else {
                return torch::zeros_like(tensor_copy);
            }
        };
        
        // Test torch::jit::fork with the created tensor
        c10::intrusive_ptr<c10::ivalue::Future> future = torch::jit::fork(async_fn);
        
        // Wait for the result and retrieve it
        torch::IValue result_ivalue = future->wait();
        if (result_ivalue.isTensor()) {
            torch::Tensor result = result_ivalue.toTensor();
            
            // Try to use the result to ensure it's valid
            if (result.defined()) {
                torch::Tensor check = result + 1;
                (void)check;
            }
        }
        
        // Test with a function that performs reduction
        torch::Tensor tensor_copy2 = input_tensor.clone();
        auto reduce_fn = [tensor_copy2]() -> torch::Tensor {
            if (tensor_copy2.numel() > 0 && tensor_copy2.dim() > 0) {
                return torch::sum(tensor_copy2, 0);
            } else {
                return tensor_copy2.clone();
            }
        };
        
        // Fork the reduction function
        c10::intrusive_ptr<c10::ivalue::Future> reduce_future = torch::jit::fork(reduce_fn);
        
        // Wait and handle potential errors
        try {
            torch::IValue reduce_result = reduce_future->wait();
            if (reduce_result.isTensor()) {
                torch::Tensor r = reduce_result.toTensor();
                (void)r;
            }
        } catch (const std::exception& e) {
            // Expected for some inputs, just continue
        }
        
        // Test with a function returning a scalar computation
        torch::Tensor tensor_copy3 = input_tensor.clone();
        auto scalar_fn = [tensor_copy3]() -> torch::Tensor {
            if (tensor_copy3.numel() > 0) {
                return torch::mean(tensor_copy3);
            } else {
                return torch::tensor(0.0f);
            }
        };
        
        c10::intrusive_ptr<c10::ivalue::Future> scalar_future = torch::jit::fork(scalar_fn);
        try {
            torch::IValue scalar_result = scalar_future->wait();
            (void)scalar_result;
        } catch (const std::exception& e) {
            // Expected for some inputs
        }
        
        // Test with multiple concurrent forks
        torch::Tensor tensor_copy4 = input_tensor.clone();
        torch::Tensor tensor_copy5 = input_tensor.clone();
        
        auto fn1 = [tensor_copy4]() -> torch::Tensor {
            return tensor_copy4 + 1;
        };
        
        auto fn2 = [tensor_copy5]() -> torch::Tensor {
            return tensor_copy5 - 1;
        };
        
        // Launch both
        c10::intrusive_ptr<c10::ivalue::Future> future1 = torch::jit::fork(fn1);
        c10::intrusive_ptr<c10::ivalue::Future> future2 = torch::jit::fork(fn2);
        
        // Wait for both
        try {
            torch::IValue res1 = future1->wait();
            torch::IValue res2 = future2->wait();
            (void)res1;
            (void)res2;
        } catch (const std::exception& e) {
            // Expected for some inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
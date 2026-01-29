#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

// Simple function to be executed by fork
torch::Tensor add_one(torch::Tensor t) {
    return t + 1;
}

// Function that uses multiple tensors
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// Function with scalar return
torch::Tensor get_numel_tensor(torch::Tensor t) {
    return torch::tensor(t.numel());
}

// Function that performs computation
torch::Tensor compute_sum(torch::Tensor t) {
    return t.sum();
}

// Function that does element-wise operation
torch::Tensor square_tensor(torch::Tensor t) {
    return t * t;
}

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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have enough data
        torch::Tensor tensor2;
        if (offset + 2 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = tensor1.clone();
        }
        
        // Get a byte to determine which test case to run
        uint8_t test_selector = 0;
        if (offset < Size) {
            test_selector = Data[offset++];
        }
        
        // Test different scenarios based on the selector using torch::jit::fork
        switch (test_selector % 5) {
            case 0: {
                // Basic fork with single tensor using add_one
                auto future = torch::jit::fork<torch::Tensor>(add_one, tensor1);
                auto result = torch::jit::wait(future);
                break;
            }
            case 1: {
                // Fork with multiple tensors
                auto future = torch::jit::fork<torch::Tensor>(add_tensors, tensor1, tensor2);
                auto result = torch::jit::wait(future);
                break;
            }
            case 2: {
                // Fork with numel computation
                auto future = torch::jit::fork<torch::Tensor>(get_numel_tensor, tensor1);
                auto result = torch::jit::wait(future);
                break;
            }
            case 3: {
                // Fork with sum computation
                auto future = torch::jit::fork<torch::Tensor>(compute_sum, tensor1);
                auto result = torch::jit::wait(future);
                break;
            }
            case 4: {
                // Fork with square operation
                auto future = torch::jit::fork<torch::Tensor>(square_tensor, tensor1);
                auto result = torch::jit::wait(future);
                break;
            }
        }
        
        // Test multiple forks in parallel
        if (offset < Size && Data[offset] % 2 == 0) {
            auto future1 = torch::jit::fork<torch::Tensor>(add_one, tensor1);
            auto future2 = torch::jit::fork<torch::Tensor>(add_one, tensor2);
            
            auto result1 = torch::jit::wait(future1);
            auto result2 = torch::jit::wait(future2);
        }
        
        // Test multiple different operations in parallel
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            auto future_add = torch::jit::fork<torch::Tensor>(add_one, tensor1);
            auto future_sum = torch::jit::fork<torch::Tensor>(compute_sum, tensor2);
            auto future_square = torch::jit::fork<torch::Tensor>(square_tensor, tensor1);
            
            auto result_add = torch::jit::wait(future_add);
            auto result_sum = torch::jit::wait(future_sum);
            auto result_square = torch::jit::wait(future_square);
        }
        
        // Test fork with lambda
        if (offset + 1 < Size && Data[offset] % 5 == 0) {
            torch::Tensor captured_tensor = tensor1.clone();
            auto future = torch::jit::fork<torch::Tensor>([](torch::Tensor t) {
                return t.abs();
            }, captured_tensor);
            auto result = torch::jit::wait(future);
        }
        
        // Test chained operations via fork
        if (offset + 2 < Size && Data[offset] % 7 == 0) {
            auto future1 = torch::jit::fork<torch::Tensor>(add_one, tensor1);
            auto intermediate = torch::jit::wait(future1);
            auto future2 = torch::jit::fork<torch::Tensor>(square_tensor, intermediate);
            auto result = torch::jit::wait(future2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
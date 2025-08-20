#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <future>
#include <thread>

// Simple function to be executed by fork
torch::Tensor add_one(torch::Tensor t) {
    return t + 1;
}

// Function that uses multiple tensors
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// Function with scalar return
int64_t get_numel(torch::Tensor t) {
    return t.numel();
}

// Function that might throw
torch::Tensor risky_operation(torch::Tensor t) {
    if (t.numel() == 0) {
        throw std::runtime_error("Empty tensor");
    }
    return t.log();
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Test different scenarios based on the selector using std::async
        switch (test_selector % 4) {
            case 0: {
                // Basic fork with single tensor
                auto future = std::async(std::launch::async, add_one, tensor1);
                auto result = future.get();
                break;
            }
            case 1: {
                // Fork with multiple tensors
                auto future = std::async(std::launch::async, add_tensors, tensor1, tensor2);
                auto result = future.get();
                break;
            }
            case 2: {
                // Fork with scalar return
                auto future = std::async(std::launch::async, get_numel, tensor1);
                auto result = future.get();
                break;
            }
            case 3: {
                // Fork with potentially throwing function
                auto future = std::async(std::launch::async, risky_operation, tensor1);
                auto result = future.get();
                break;
            }
        }
        
        // Test multiple forks in parallel
        if (offset < Size && Data[offset] % 2 == 0) {
            auto future1 = std::async(std::launch::async, add_one, tensor1);
            auto future2 = std::async(std::launch::async, add_one, tensor2);
            
            auto result1 = future1.get();
            auto result2 = future2.get();
        }
        
        // Test nested forks
        if (offset < Size && Data[offset] % 3 == 0) {
            auto outer_future = std::async(std::launch::async, [&tensor1, &tensor2]() {
                auto inner_future = std::async(std::launch::async, add_tensors, tensor1, tensor2);
                return inner_future.get();
            });
            auto result = outer_future.get();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
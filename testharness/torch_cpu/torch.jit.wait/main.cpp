#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <future>
#include <thread>
#include <chrono>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple future that returns the tensor
        auto future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
        future->markCompleted(tensor);
        
        // Test future wait functionality
        torch::Tensor result = future->wait().toTensor();
        
        // Try with a future that's not yet completed
        if (offset < Size) {
            auto delayed_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            
            // Create a second tensor if we have more data
            torch::Tensor second_tensor;
            if (Size - offset > 2) {
                second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                second_tensor = tensor.clone();
            }
            
            // Complete the future after a small delay
            std::thread([delayed_future, second_tensor]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                delayed_future->markCompleted(second_tensor);
            }).detach();
            
            // Wait for the future to complete
            torch::Tensor delayed_result = delayed_future->wait().toTensor();
        }
        
        // Test with timeout
        if (offset < Size && Size - offset >= 1) {
            double timeout_seconds = static_cast<double>(Data[offset++]) / 255.0;
            
            auto timeout_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            
            // Try waiting with timeout
            try {
                timeout_future->waitAndThrow(std::chrono::duration<double>(timeout_seconds));
            } catch (const c10::Error& e) {
                // Expected timeout exception
            }
            
            // Complete the future and try again
            timeout_future->markCompleted(tensor);
            torch::Tensor timeout_result = timeout_future->wait().toTensor();
        }
        
        // Test with a future that contains an error
        if (offset < Size) {
            auto error_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            try {
                error_future->setError(std::make_exception_ptr(std::runtime_error("Test error from fuzzer")));
                error_future->wait();
            } catch (const c10::Error& e) {
                // Expected error
            } catch (const std::exception& e) {
                // Expected error
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

#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <thread>
#include <chrono>
#include <atomic>

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
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple future that returns the tensor
        auto future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
        future->markCompleted(tensor);
        
        // Test future wait functionality - wait() returns void, use value() to get result
        future->wait();
        torch::Tensor result = future->value().toTensor();
        
        // Test with a future that's completed synchronously
        if (offset < Size) {
            auto sync_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            
            // Create a second tensor if we have more data
            torch::Tensor second_tensor;
            if (Size - offset > 2) {
                second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                second_tensor = tensor.clone();
            }
            
            // Complete the future immediately (avoid threading issues in fuzzer)
            sync_future->markCompleted(second_tensor);
            
            // Wait for the future to complete, then get value
            sync_future->wait();
            torch::Tensor sync_result = sync_future->value().toTensor();
            (void)sync_result;
        }
        
        // Test with a future completed from a thread (properly joined)
        if (offset < Size && Size - offset >= 1) {
            auto threaded_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            torch::Tensor tensor_copy = tensor.clone();
            
            std::thread completion_thread([threaded_future, tensor_copy]() {
                threaded_future->markCompleted(tensor_copy);
            });
            
            // Wait for the future (blocks until completed), then get value
            threaded_future->wait();
            torch::Tensor threaded_result = threaded_future->value().toTensor();
            (void)threaded_result;
            
            // Join the thread to ensure proper cleanup
            completion_thread.join();
        }
        
        // Test with a future that contains an error
        if (offset < Size) {
            auto error_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            try {
                error_future->setError(std::make_exception_ptr(std::runtime_error("Test error from fuzzer")));
                error_future->wait();
                // Accessing value() after error should throw
                error_future->value();
            } catch (const c10::Error& e) {
                // Expected error
            } catch (const std::runtime_error& e) {
                // Expected error
            }
        }
        
        // Test future with different IValue types
        if (offset < Size) {
            // Test with int
            auto int_future = c10::make_intrusive<c10::ivalue::Future>(c10::IntType::get());
            int64_t int_val = static_cast<int64_t>(Data[offset % Size]);
            int_future->markCompleted(int_val);
            int_future->wait();
            int64_t int_result = int_future->value().toInt();
            (void)int_result;
            
            // Test with double
            auto double_future = c10::make_intrusive<c10::ivalue::Future>(c10::FloatType::get());
            double double_val = static_cast<double>(Data[offset % Size]) / 255.0;
            double_future->markCompleted(double_val);
            double_future->wait();
            double double_result = double_future->value().toDouble();
            (void)double_result;
            
            // Test with bool
            auto bool_future = c10::make_intrusive<c10::ivalue::Future>(c10::BoolType::get());
            bool bool_val = Data[offset % Size] > 127;
            bool_future->markCompleted(bool_val);
            bool_future->wait();
            bool bool_result = bool_future->value().toBool();
            (void)bool_result;
        }
        
        // Test hasValue and completed
        {
            auto check_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            bool completed_before = check_future->completed();
            check_future->markCompleted(tensor);
            bool completed_after = check_future->completed();
            (void)completed_before;
            (void)completed_after;
        }
        
        (void)result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
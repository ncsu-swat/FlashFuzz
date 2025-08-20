#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/torch.h>
#include <future>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Future object
        auto future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
        
        // Set value to the future
        future->markCompleted(c10::IValue(input_tensor));
        
        // Test wait() functionality
        c10::IValue result = future->wait();
        
        // Test value() functionality
        c10::IValue value_result = future->value();
        
        // Test completed() functionality
        bool is_completed = future->completed();
        
        // Test hasValue() functionality
        bool has_value = future->hasValue();
        
        // Test hasError() functionality
        bool has_error = future->hasError();
        
        // Create a Future with error
        if (offset < Size) {
            auto error_future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
            std::string error_msg = "Test error message";
            error_future->setError(std::make_exception_ptr(std::runtime_error(error_msg)));
            
            // Test hasError() on error future
            bool error_has_error = error_future->hasError();
            
            // Test tryRetrieveErrorMessage()
            if (error_has_error) {
                std::string retrieved_error = error_future->tryRetrieveErrorMessage();
            }
            
            try {
                // This should throw
                c10::IValue error_result = error_future->value();
            } catch (const std::exception& e) {
                // Expected exception
            }
        }
        
        // Test then() functionality
        if (offset < Size) {
            auto then_future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
            auto next_future = then_future->then([](c10::IValue val) {
                torch::Tensor t = val.toTensor();
                return c10::IValue(t * 2);
            }, c10::TensorType::get());
            
            then_future->markCompleted(c10::IValue(input_tensor));
            c10::IValue then_result = next_future->wait();
        }
        
        // Test Future with different types
        if (offset < Size) {
            auto int_future = c10::make_intrusive<torch::jit::Future>(c10::IntType::get());
            int value = static_cast<int>(Data[offset % Size]);
            int_future->markCompleted(c10::IValue(value));
            c10::IValue int_result = int_future->wait();
        }
        
        // Test Future with void type
        if (offset < Size) {
            auto void_future = c10::make_intrusive<torch::jit::Future>(c10::NoneType::get());
            void_future->markCompleted(c10::IValue());
            void_future->wait();
        }
        
        // Test Future with list type
        if (offset < Size) {
            auto list_type = c10::ListType::create(c10::TensorType::get());
            auto list_future = c10::make_intrusive<torch::jit::Future>(list_type);
            c10::List<torch::Tensor> tensors;
            tensors.push_back(input_tensor);
            tensors.push_back(input_tensor);
            list_future->markCompleted(c10::IValue(tensors));
            c10::IValue list_result = list_future->wait();
        }
        
        // Test Future with tuple type
        if (offset < Size) {
            std::vector<c10::TypePtr> tuple_types = {c10::TensorType::get(), c10::IntType::get()};
            auto tuple_type = c10::TupleType::create(tuple_types);
            auto tuple_future = c10::make_intrusive<torch::jit::Future>(tuple_type);
            std::vector<c10::IValue> tuple_elements = {c10::IValue(input_tensor), c10::IValue(42)};
            tuple_future->markCompleted(c10::IValue(c10::ivalue::Tuple::create(tuple_elements)));
            c10::IValue tuple_result = tuple_future->wait();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
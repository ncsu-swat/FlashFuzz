#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use remaining data to determine which code paths to exercise
        uint8_t path_selector = (offset < Size) ? Data[offset++] : 0;
        
        // Create a Future object
        auto future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
        
        // Set value to the future
        future->markCompleted(c10::IValue(input_tensor));
        
        // Test wait() functionality
        future->wait();
        
        // Test value() functionality
        c10::IValue value_result = future->value();
        
        // Test completed() functionality
        bool is_completed = future->completed();
        (void)is_completed;
        
        // Test hasValue() functionality
        bool has_value = future->hasValue();
        (void)has_value;
        
        // Test hasError() functionality
        bool has_error = future->hasError();
        (void)has_error;
        
        // Create a Future with error
        if (path_selector & 0x01) {
            auto error_future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
            std::string error_msg = "Test error message";
            error_future->setError(std::make_exception_ptr(std::runtime_error(error_msg)));
            
            // Test hasError() on error future
            bool error_has_error = error_future->hasError();
            
            // Test tryRetrieveErrorMessage()
            if (error_has_error) {
                std::string retrieved_error = error_future->tryRetrieveErrorMessage();
                (void)retrieved_error;
            }
            
            try {
                // This should throw
                c10::IValue error_result = error_future->value();
                (void)error_result;
            } catch (const std::exception&) {
                // Expected exception - silently catch
            }
        }
        
        // Test then() functionality
        if (path_selector & 0x02) {
            auto then_future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
            auto next_future = then_future->then(
                [](torch::jit::Future& parent_future) -> c10::IValue {
                    torch::Tensor t = parent_future.constValue().toTensor();
                    return c10::IValue(t * 2);
                }, 
                c10::TensorType::get()
            );
            
            then_future->markCompleted(c10::IValue(input_tensor));
            next_future->wait();
            c10::IValue then_result = next_future->value();
            (void)then_result;
        }
        
        // Test Future with different types - int
        if (path_selector & 0x04) {
            auto int_future = c10::make_intrusive<torch::jit::Future>(c10::IntType::get());
            int64_t value = static_cast<int64_t>(offset < Size ? Data[offset++] : 42);
            int_future->markCompleted(c10::IValue(value));
            int_future->wait();
            c10::IValue int_result = int_future->value();
            (void)int_result;
        }
        
        // Test Future with None type
        if (path_selector & 0x08) {
            auto none_future = c10::make_intrusive<torch::jit::Future>(c10::NoneType::get());
            none_future->markCompleted(c10::IValue());
            none_future->wait();
            c10::IValue none_result = none_future->value();
            (void)none_result;
        }
        
        // Test Future with list type
        if (path_selector & 0x10) {
            auto list_type = c10::ListType::create(c10::TensorType::get());
            auto list_future = c10::make_intrusive<torch::jit::Future>(list_type);
            c10::List<torch::Tensor> tensors;
            tensors.push_back(input_tensor);
            tensors.push_back(input_tensor.clone());
            list_future->markCompleted(c10::IValue(tensors));
            list_future->wait();
            c10::IValue list_result = list_future->value();
            (void)list_result;
        }
        
        // Test Future with tuple type
        if (path_selector & 0x20) {
            std::vector<c10::TypePtr> tuple_types = {c10::TensorType::get(), c10::IntType::get()};
            auto tuple_type = c10::TupleType::create(tuple_types);
            auto tuple_future = c10::make_intrusive<torch::jit::Future>(tuple_type);
            int64_t int_val = (offset < Size) ? static_cast<int64_t>(Data[offset++]) : 42;
            std::vector<c10::IValue> tuple_elements = {c10::IValue(input_tensor), c10::IValue(int_val)};
            tuple_future->markCompleted(c10::IValue(c10::ivalue::Tuple::create(tuple_elements)));
            tuple_future->wait();
            c10::IValue tuple_result = tuple_future->value();
            (void)tuple_result;
        }
        
        // Test Future with bool type
        if (path_selector & 0x40) {
            auto bool_future = c10::make_intrusive<torch::jit::Future>(c10::BoolType::get());
            bool bool_val = (offset < Size) ? (Data[offset++] & 0x01) : true;
            bool_future->markCompleted(c10::IValue(bool_val));
            bool_future->wait();
            c10::IValue bool_result = bool_future->value();
            (void)bool_result;
        }
        
        // Test Future with double type
        if (path_selector & 0x80) {
            auto float_future = c10::make_intrusive<torch::jit::Future>(c10::FloatType::get());
            double float_val = (offset < Size) ? static_cast<double>(Data[offset++]) / 255.0 : 0.5;
            float_future->markCompleted(c10::IValue(float_val));
            float_future->wait();
            c10::IValue float_result = float_future->value();
            (void)float_result;
        }
        
        // Test addCallback functionality
        if (offset < Size && (Data[offset++] & 0x01)) {
            auto callback_future = c10::make_intrusive<torch::jit::Future>(c10::TensorType::get());
            bool callback_called = false;
            callback_future->addCallback([&callback_called](torch::jit::Future& f) {
                callback_called = true;
                (void)f;
            });
            callback_future->markCompleted(c10::IValue(input_tensor));
            (void)callback_called;
        }
        
        // Test string type
        if (offset < Size) {
            auto string_future = c10::make_intrusive<torch::jit::Future>(c10::StringType::get());
            size_t str_len = std::min(static_cast<size_t>(Data[offset++] % 64), Size - offset);
            std::string str_val(reinterpret_cast<const char*>(Data + offset), str_len);
            offset += str_len;
            string_future->markCompleted(c10::IValue(str_val));
            string_future->wait();
            c10::IValue string_result = string_future->value();
            (void)string_result;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
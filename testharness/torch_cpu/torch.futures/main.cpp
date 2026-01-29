#include "fuzzer_utils.h"
#include <iostream>
#include <chrono>

// Target API: torch.futures (c10::ivalue::Future)

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
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a future with TensorType
        auto future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
        
        // Get a byte to determine the test case
        uint8_t test_case = 0;
        if (offset < Size) {
            test_case = Data[offset++];
        }
        
        // Test different future operations based on the test case
        switch (test_case % 5) {
            case 0: {
                // Test immediate setting of value
                future->markCompleted(tensor);
                if (future->completed()) {
                    auto result = future->value().toTensor();
                    // Verify tensor is valid
                    (void)result.numel();
                }
                break;
            }
            case 1: {
                // Test markCompleted and wait (synchronous, no threads)
                future->markCompleted(tensor);
                future->wait();
                if (future->completed()) {
                    auto result = future->value().toTensor();
                    (void)result.numel();
                }
                break;
            }
            case 2: {
                // Test then() callback
                future->markCompleted(tensor);
                auto chained = future->then(
                    [](c10::ivalue::Future& parent) -> c10::IValue {
                        return parent.value();
                    },
                    c10::TensorType::get()
                );
                chained->wait();
                if (chained->completed()) {
                    auto result = chained->value().toTensor();
                    (void)result.numel();
                }
                break;
            }
            case 3: {
                // Test setting error
                try {
                    std::string error_msg = "Test error from fuzzer";
                    future->setError(std::make_exception_ptr(std::runtime_error(error_msg)));
                    
                    // This should throw when we try to get value
                    future->wait();
                    // Try to access value - this should throw
                    (void)future->value();
                } catch (const std::exception&) {
                    // Expected exception - silently catch
                }
                break;
            }
            case 4: {
                // Test hasValue and completed states
                if (!future->completed()) {
                    future->markCompleted(tensor);
                }
                bool has_val = future->hasValue();
                bool is_complete = future->completed();
                (void)has_val;
                (void)is_complete;
                break;
            }
        }
        
        // Test creating multiple futures and completing them
        if (offset + 4 < Size) {
            uint8_t num_futures = Data[offset++] % 5 + 1;  // 1-5 futures
            
            std::vector<c10::intrusive_ptr<c10::ivalue::Future>> future_vec;
            
            for (uint8_t i = 0; i < num_futures && offset < Size; i++) {
                auto new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto new_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
                new_future->markCompleted(new_tensor);
                future_vec.push_back(new_future);
            }
            
            // Verify all futures completed
            for (const auto& f : future_vec) {
                if (f->completed()) {
                    auto result = f->value().toTensor();
                    (void)result.numel();
                }
            }
        }
        
        // Test chaining multiple then() calls
        if (offset + 2 < Size) {
            auto chain_future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            auto chain_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            uint8_t chain_depth = Data[offset % Size] % 3 + 1;  // 1-3 chain depth
            
            chain_future->markCompleted(chain_tensor);
            
            auto current = chain_future;
            for (uint8_t i = 0; i < chain_depth; i++) {
                current = current->then(
                    [](c10::ivalue::Future& parent) -> c10::IValue {
                        auto t = parent.value().toTensor();
                        // Simple transformation: clone the tensor
                        return t.clone();
                    },
                    c10::TensorType::get()
                );
            }
            
            current->wait();
            if (current->completed()) {
                auto final_result = current->value().toTensor();
                (void)final_result.numel();
            }
        }
        
        // Test Future with different IValue types
        if (offset < Size) {
            uint8_t type_case = Data[offset++] % 3;
            
            switch (type_case) {
                case 0: {
                    // Int future
                    auto int_future = c10::make_intrusive<c10::ivalue::Future>(c10::IntType::get());
                    int64_t val = static_cast<int64_t>(offset < Size ? Data[offset] : 42);
                    int_future->markCompleted(c10::IValue(val));
                    int_future->wait();
                    if (int_future->completed()) {
                        auto result = int_future->value().toInt();
                        (void)result;
                    }
                    break;
                }
                case 1: {
                    // Double future
                    auto double_future = c10::make_intrusive<c10::ivalue::Future>(c10::FloatType::get());
                    double val = static_cast<double>(offset < Size ? Data[offset] : 3) / 10.0;
                    double_future->markCompleted(c10::IValue(val));
                    double_future->wait();
                    if (double_future->completed()) {
                        auto result = double_future->value().toDouble();
                        (void)result;
                    }
                    break;
                }
                case 2: {
                    // Bool future
                    auto bool_future = c10::make_intrusive<c10::ivalue::Future>(c10::BoolType::get());
                    bool val = (offset < Size) ? (Data[offset] % 2 == 0) : true;
                    bool_future->markCompleted(c10::IValue(val));
                    bool_future->wait();
                    if (bool_future->completed()) {
                        auto result = bool_future->value().toBool();
                        (void)result;
                    }
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
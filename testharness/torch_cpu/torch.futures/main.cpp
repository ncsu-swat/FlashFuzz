#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <thread>         // For std::thread
#include <chrono>         // For std::chrono

// Target API: torch.futures
// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a future
        auto future = torch::make_intrusive<torch::ivalue::Future>(c10::TensorType::get());
        
        // Get a byte to determine the test case
        uint8_t test_case = 0;
        if (offset < Size) {
            test_case = Data[offset++];
        }
        
        // Test different future operations based on the test case
        switch (test_case % 4) {
            case 0: {
                // Test immediate setting of value
                future->markCompleted(tensor);
                if (future->completed()) {
                    auto result = future->value().toTensor();
                }
                break;
            }
            case 1: {
                // Test setting value after a delay
                std::thread([future, tensor]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    future->markCompleted(tensor);
                }).detach();
                
                future->wait();
                break;
            }
            case 2: {
                // Test then() callback
                future->markCompleted(tensor);
                future->then([](c10::ivalue::Future &parent) {
                    return parent.value();
                }, c10::TensorType::get());
                break;
            }
            case 3: {
                // Test setting error
                if (offset < Size) {
                    try {
                        std::string error_msg = "Test error";
                        future->setError(std::make_exception_ptr(std::runtime_error(error_msg)));
                        
                        // This should throw
                        future->wait();
                    } catch (const std::exception& e) {
                        // Expected exception
                    }
                } else {
                    future->markCompleted(tensor);
                    future->wait();
                }
                break;
            }
        }
        
        // Test collectAll if we have enough data left
        if (offset + 4 < Size) {
            c10::List<c10::intrusive_ptr<torch::ivalue::Future>> futures(c10::AnyType::get());
            
            // Create a few more tensors and futures
            uint8_t num_futures = Data[offset++] % 5 + 1;  // 1-5 futures
            
            for (uint8_t i = 0; i < num_futures && offset < Size; i++) {
                auto new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto new_future = torch::make_intrusive<torch::ivalue::Future>(c10::TensorType::get());
                new_future->markCompleted(new_tensor);
                futures.push_back(new_future);
            }
            
            // Test collectAll
            auto collected_future = torch::collectAll(futures);
            collected_future->wait();
        }
        
        // Test collectAny if we have enough data left
        if (offset + 4 < Size) {
            c10::List<c10::intrusive_ptr<torch::ivalue::Future>> futures(c10::AnyType::get());
            
            // Create a few more tensors and futures
            uint8_t num_futures = Data[offset++] % 5 + 1;  // 1-5 futures
            
            for (uint8_t i = 0; i < num_futures && offset < Size; i++) {
                auto new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto new_future = torch::make_intrusive<torch::ivalue::Future>(c10::TensorType::get());
                new_future->markCompleted(new_tensor);
                futures.push_back(new_future);
            }
            
            // Test collectAny
            auto collected_future = torch::collectAny(futures);
            collected_future->wait();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

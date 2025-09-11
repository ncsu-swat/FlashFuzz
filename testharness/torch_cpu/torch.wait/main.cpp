#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Create a list of futures to test torch::jit::wait
        std::vector<c10::intrusive_ptr<c10::ivalue::Future>> futures;
        
        // Create 1-3 futures based on available data
        int num_futures = (Size > 0) ? (Data[0] % 3) + 1 : 1;
        offset++;
        
        for (int i = 0; i < num_futures && offset < Size; i++) {
            // Create a tensor to use in the future
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a future that will be immediately completed with the tensor
            auto future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            future->markCompleted(tensor);
            
            // Add the future to our list
            futures.push_back(future);
        }
        
        // If we have no futures, create at least one
        if (futures.empty()) {
            auto future = c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
            future->markCompleted(torch::ones({1}));
            futures.push_back(future);
        }
        
        // Test different timeout values
        double timeout_sec = 0.0;
        if (offset < Size) {
            // Use some bytes to determine timeout
            if (Size - offset >= sizeof(double)) {
                std::memcpy(&timeout_sec, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else {
                timeout_sec = static_cast<double>(Data[offset++]) / 1000.0;
            }
        }
        
        // Call torch::jit::wait with the futures
        auto result = torch::jit::wait(futures, timeout_sec);
        
        // Test the returned values
        for (const auto& completed_future : result) {
            if (completed_future->completed()) {
                auto value = completed_future->value();
                if (value.isTensor()) {
                    auto tensor = value.toTensor();
                    
                    // Do something with the tensor to ensure it's valid
                    auto sum = tensor.sum();
                    
                    // Try to access tensor properties
                    auto sizes = tensor.sizes();
                    auto dtype = tensor.dtype();
                    auto device = tensor.device();
                }
            }
        }
        
        // Test with empty list of futures
        std::vector<c10::intrusive_ptr<c10::ivalue::Future>> empty_futures;
        auto empty_result = torch::jit::wait(empty_futures, 0.0);
        
        // Test with negative timeout
        if (offset < Size && Data[offset] % 2 == 0) {
            auto neg_result = torch::jit::wait(futures, -1.0);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

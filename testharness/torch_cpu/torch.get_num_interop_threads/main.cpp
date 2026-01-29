#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Get the number of interop threads
        // This is a simple getter that returns the current thread pool size
        int num_threads = torch::get_num_interop_threads();
        
        // Verify the returned value is sane (should be >= 1)
        if (num_threads < 1) {
            std::cerr << "Unexpected: num_interop_threads < 1: " << num_threads << std::endl;
        }
        
        // Note: torch::set_num_interop_threads() can only be called ONCE
        // before any inter-op parallel work is done. It cannot be meaningfully
        // fuzzed as calling it multiple times throws an exception by design.
        // Therefore, we only test the getter here.
        
        // Use the fuzz data to create tensors and exercise operations that
        // may use the interop thread pool internally
        if (Size > 0) {
            size_t offset = 0;
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform operations that may use interop threads internally
                // Matrix multiplication can use multiple threads
                if (tensor.dim() >= 2 && tensor.size(-1) > 0 && tensor.size(-2) > 0) {
                    auto t = tensor.select(0, 0).unsqueeze(0);
                    if (t.dim() == 2) {
                        torch::mm(t, t.transpose(0, 1));
                    }
                }
                
                // Reduction operations may also use threading
                torch::sum(tensor);
                torch::mean(tensor.to(torch::kFloat));
                
            } catch (const std::exception &) {
                // Shape/type mismatches are expected with random data
            }
        }
        
        // Call get_num_interop_threads again to ensure it's stable
        int num_threads_after = torch::get_num_interop_threads();
        
        // Sanity check: thread count shouldn't change during execution
        if (num_threads != num_threads_after) {
            std::cerr << "Thread count changed unexpectedly: " 
                      << num_threads << " -> " << num_threads_after << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
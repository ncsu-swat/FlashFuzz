#include "fuzzer_utils.h"
#include <iostream>
#include <atomic>

// Flag to track if we've already set interop threads (can only be done once)
static std::atomic<bool> threads_already_set{false};
static std::atomic<bool> first_attempt_done{false};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 1) {
            return 0;
        }

        size_t offset = 0;

        // Get current interop threads (this is always safe to call)
        int current_threads = torch::get_num_interop_threads();

        // set_num_interop_threads can only be called once before any parallel work
        // After the first successful call, subsequent calls will throw
        if (!first_attempt_done.exchange(true)) {
            // First iteration: try to set the number of threads
            // Use the fuzzer input to determine the thread count
            int num_threads = static_cast<int>(Data[0] % 16) + 1; // 1-16 threads
            
            try {
                torch::set_num_interop_threads(num_threads);
                threads_already_set.store(true);
                
                // Verify it was set
                int new_threads = torch::get_num_interop_threads();
                if (new_threads != num_threads) {
                    std::cerr << "Warning: requested " << num_threads 
                              << " threads but got " << new_threads << std::endl;
                }
            } catch (const std::exception &e) {
                // Setting threads may fail if parallel work already started
                // (e.g., from library initialization)
            }
        }
        
        offset = 1;

        // The main value of this fuzzer after initial setup is to exercise
        // operations that use the interop thread pool
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (tensor.defined() && tensor.numel() > 0) {
                // Operations that may use interop parallelism
                try {
                    // Matrix operations can use parallelism
                    if (tensor.dim() >= 2) {
                        auto t = tensor.to(torch::kFloat);
                        auto result = torch::mm(t.view({-1, t.size(-1)}), 
                                               t.view({t.size(-1), -1}));
                        volatile float sum = result.sum().item<float>();
                        (void)sum;
                    } else {
                        // Simple operations
                        auto result = tensor.clone();
                        volatile float sum = result.sum().item<float>();
                        (void)sum;
                    }
                } catch (const std::exception &) {
                    // Shape mismatches etc. are expected
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
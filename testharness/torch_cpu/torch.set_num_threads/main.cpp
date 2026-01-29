#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least 1 byte for the number of threads
        if (Size < 1) {
            return 0;
        }
        
        // Extract number of threads from fuzzer data
        // Map to a reasonable range: 1 to 64 threads (valid values)
        int num_threads = (static_cast<int>(Data[0]) % 64) + 1;
        offset++;
        
        // Set the number of threads
        torch::set_num_threads(num_threads);
        
        // Verify the setting by getting current threads
        int current_threads = torch::get_num_threads();
        
        // Create a tensor and perform operations that can utilize multiple threads
        if (Size > offset) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform computations that could potentially use multiple threads
            // These are parallelizable operations in PyTorch
            torch::Tensor result = tensor.sum();
            
            // Matrix operations tend to be parallelized
            if (tensor.dim() >= 2) {
                try {
                    torch::Tensor result2 = tensor.mean(0);
                    torch::Tensor result3 = tensor.std(0);
                } catch (...) {
                    // Shape-related issues, ignore silently
                }
            }
            
            // Element-wise operations
            torch::Tensor exp_result = torch::exp(tensor);
            torch::Tensor sin_result = torch::sin(tensor);
            
            // Try a matmul if we have enough dimensions
            if (tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0) {
                try {
                    torch::Tensor transposed = tensor.t();
                    torch::Tensor matmul_result = torch::matmul(tensor, transposed);
                } catch (...) {
                    // Shape mismatch, ignore silently
                }
            }
        }
        
        // Test changing thread count mid-execution
        if (Size > offset) {
            int new_threads = (static_cast<int>(Data[offset]) % 32) + 1;
            offset++;
            torch::set_num_threads(new_threads);
            
            // Perform another operation with different thread count
            if (Size > offset) {
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result = tensor2.sum();
            }
        }
        
        // Test with interop threads as well
        if (Size > offset) {
            int interop_threads = (static_cast<int>(Data[offset]) % 16) + 1;
            offset++;
            torch::set_num_interop_threads(interop_threads);
            int current_interop = torch::get_num_interop_threads();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
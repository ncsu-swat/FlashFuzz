#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the atan_ operation in-place
        // atan_ computes arctangent element-wise in-place
        tensor.atan_();
        
        // Ensure the result is computed (avoid lazy evaluation issues)
        if (tensor.defined()) {
            // Access a value to ensure computation happened
            (void)tensor.numel();
        }
        
        // Try different tensor configurations if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2.atan_();
        }
        
        // Test with a contiguous tensor
        if (offset + 2 < Size) {
            torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            if (!tensor3.is_contiguous()) {
                tensor3 = tensor3.contiguous();
            }
            tensor3.atan_();
        }
        
        // Test atan_ on a slice/view if possible
        if (offset + 2 < Size) {
            torch::Tensor tensor4 = fuzzer_utils::createTensor(Data, Size, offset);
            if (tensor4.numel() > 1) {
                try {
                    // Create a view and apply atan_ to it
                    auto slice = tensor4.slice(0, 0, tensor4.size(0) > 1 ? tensor4.size(0) / 2 : 1);
                    slice.atan_();
                } catch (...) {
                    // Slicing may fail for certain tensor shapes, ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor - either identical or slightly different
        torch::Tensor tensor2;
        
        // Use a byte to determine if tensors should be equal or not
        if (offset < Size) {
            uint8_t should_be_equal = Data[offset++];
            
            if (should_be_equal % 2 == 0) {
                // Make identical tensor
                tensor2 = tensor1.clone();
            } else {
                // Create a different tensor
                if (offset < Size) {
                    // Create completely new tensor
                    tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    // Modify the first tensor slightly if we can
                    tensor2 = tensor1.clone();
                    if (tensor2.numel() > 0 && !tensor2.is_meta()) {
                        // Try to modify one element if possible
                        if (tensor2.dtype() == torch::kBool) {
                            // For boolean tensors, flip a value
                            if (tensor2.numel() > 0) {
                                tensor2.index_put_({0}, !tensor2.index({0}).item<bool>());
                            }
                        } else {
                            // For numeric tensors, add a small value
                            tensor2 = tensor2 + 1;
                        }
                    }
                }
            }
        } else {
            // Not enough data, just clone the first tensor
            tensor2 = tensor1.clone();
        }
        
        // Test torch::equal - use volatile to prevent optimization
        volatile bool are_equal = torch::equal(tensor1, tensor2);
        (void)are_equal;
        
        // Test the reverse order too (should give same result)
        volatile bool are_equal_reversed = torch::equal(tensor2, tensor1);
        (void)are_equal_reversed;
        
        // Test with self (should always be true)
        volatile bool self_equal = torch::equal(tensor1, tensor1);
        (void)self_equal;
        
        // Edge case: test with empty tensors
        try {
            torch::Tensor empty_tensor1 = torch::empty({0});
            torch::Tensor empty_tensor2 = torch::empty({0});
            volatile bool empty_equal = torch::equal(empty_tensor1, empty_tensor2);
            (void)empty_equal;
        } catch (...) {
            // Ignore failures on edge cases
        }
        
        // Edge case: test with tensors of different dtypes but same values
        try {
            if (tensor1.numel() > 0 && tensor1.dtype() != torch::kBool && 
                tensor1.dtype() != torch::kComplexFloat && tensor1.dtype() != torch::kComplexDouble) {
                torch::Tensor tensor1_float = tensor1.to(torch::kFloat);
                torch::Tensor tensor1_int = tensor1.to(torch::kInt);
                volatile bool diff_dtype_equal = torch::equal(tensor1_float, tensor1_int);
                (void)diff_dtype_equal;
            }
        } catch (...) {
            // Ignore type conversion failures
        }
        
        // Edge case: test with tensors of different shapes
        try {
            if (tensor1.dim() > 0 && tensor1.size(0) > 1) {
                torch::Tensor reshaped = tensor1.reshape({-1});
                volatile bool diff_shape_equal = torch::equal(tensor1, reshaped);
                (void)diff_shape_equal;
            }
        } catch (...) {
            // Ignore reshape failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
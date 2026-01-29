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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // lcm only works with integer types, convert if necessary
        if (tensor1.is_floating_point() || tensor1.is_complex()) {
            tensor1 = tensor1.to(torch::kInt64);
        }
        
        // Create second input tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (tensor2.is_floating_point() || tensor2.is_complex()) {
                tensor2 = tensor2.to(torch::kInt64);
            }
        } else {
            // If no data left, create a simple integer tensor
            tensor2 = torch::randint(1, 10, tensor1.sizes(), torch::kInt64);
        }
        
        // Variant 1: lcm with two tensors
        try {
            torch::Tensor result1 = torch::lcm(tensor1, tensor2);
        } catch (...) {
            // Shape mismatch or other expected failures
        }
        
        // Variant 2: lcm with scalar tensor
        if (Size > offset) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset % Size]) + 1; // +1 to avoid 0
            offset++;
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, torch::kInt64);
            
            try {
                torch::Tensor result2 = torch::lcm(tensor1, scalar_tensor);
            } catch (...) {
                // Expected failures
            }
            
            try {
                torch::Tensor result3 = torch::lcm(scalar_tensor, tensor1);
            } catch (...) {
                // Expected failures
            }
        }
        
        // Variant 3: out variant
        try {
            torch::Tensor out_tensor = torch::empty_like(tensor1);
            torch::lcm_out(out_tensor, tensor1, tensor2);
        } catch (...) {
            // Shape mismatch or dtype issues
        }
        
        // Variant 4: in-place lcm (only works with integer types)
        try {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.lcm_(tensor2);
        } catch (...) {
            // Expected failures from shape mismatch
        }
        
        // Variant 5: lcm with explicit broadcasting via expand
        try {
            if (tensor1.dim() > 0 && tensor1.numel() > 0) {
                // Create a 1D tensor and try to broadcast
                torch::Tensor broadcast_tensor = torch::tensor({2, 3, 5}, torch::kInt64);
                torch::Tensor result_broadcast = torch::lcm(tensor1.flatten(), broadcast_tensor);
            }
        } catch (...) {
            // Broadcasting failures are expected
        }
        
        // Variant 6: lcm with negative values
        try {
            torch::Tensor neg_tensor = tensor1 * -1;
            torch::Tensor result_neg = torch::lcm(tensor1, neg_tensor);
        } catch (...) {
            // Handle any issues with negative values
        }
        
        // Variant 7: lcm with zeros (edge case)
        try {
            torch::Tensor zero_tensor = torch::zeros_like(tensor1);
            torch::Tensor result_zero = torch::lcm(tensor1, zero_tensor);
        } catch (...) {
            // lcm with zero may have special behavior
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
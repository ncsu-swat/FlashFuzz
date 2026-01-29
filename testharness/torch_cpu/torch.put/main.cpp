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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create the destination tensor (the tensor to modify)
        torch::Tensor destination = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get total number of elements in destination for valid index generation
        int64_t numel = destination.numel();
        if (numel == 0) {
            return 0;
        }
        
        // Create index tensor with valid indices
        torch::Tensor raw_index;
        if (offset < Size) {
            raw_index = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            raw_index = torch::zeros({1}, torch::kFloat);
        }
        
        // Convert to long and ensure indices are valid (within bounds)
        torch::Tensor index;
        try {
            // Flatten and convert to long
            index = raw_index.flatten().to(torch::kLong);
            // Make indices valid by taking modulo of numel (handles negative indices too)
            index = torch::remainder(index, numel);
            // Ensure non-negative
            index = torch::where(index < 0, index + numel, index);
        } catch (...) {
            // Fallback to simple valid index
            index = torch::zeros({1}, torch::kLong);
        }
        
        // Create values tensor with matching number of elements
        torch::Tensor values;
        if (offset < Size) {
            torch::Tensor raw_values = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Flatten and resize to match index size
                values = raw_values.flatten();
                int64_t idx_numel = index.numel();
                if (values.numel() >= idx_numel) {
                    values = values.slice(0, 0, idx_numel);
                } else {
                    // Repeat values to match index size
                    int64_t repeats = (idx_numel / values.numel()) + 1;
                    values = values.repeat({repeats}).slice(0, 0, idx_numel);
                }
                // Match dtype with destination
                values = values.to(destination.scalar_type());
            } catch (...) {
                values = torch::ones({index.numel()}, destination.options());
            }
        } else {
            values = torch::ones({index.numel()}, destination.options());
        }
        
        // Determine accumulate flag from fuzzer data
        bool accumulate = false;
        if (offset < Size) {
            accumulate = Data[offset++] % 2 == 1;
        }
        
        // Variant 1: Basic put_ without accumulate
        try {
            torch::Tensor result1 = destination.clone();
            result1.put_(index, values);
            // Force computation
            (void)result1.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
        // Variant 2: put_ with accumulate=false
        try {
            torch::Tensor result2 = destination.clone();
            result2.put_(index, values, false);
            (void)result2.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
        // Variant 3: put_ with accumulate=true
        try {
            torch::Tensor result3 = destination.clone();
            result3.put_(index, values, true);
            (void)result3.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
        // Variant 4: put_ with fuzzer-controlled accumulate flag
        try {
            torch::Tensor result4 = destination.clone();
            result4.put_(index, values, accumulate);
            (void)result4.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
        // Variant 5: Using contiguous tensors
        try {
            torch::Tensor result5 = destination.clone().contiguous();
            torch::Tensor cont_index = index.contiguous();
            torch::Tensor cont_values = values.contiguous();
            result5.put_(cont_index, cont_values, accumulate);
            (void)result5.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
        // Variant 6: Single element put
        try {
            torch::Tensor result6 = destination.clone();
            torch::Tensor single_idx = torch::tensor({0}, torch::kLong);
            torch::Tensor single_val = torch::tensor({1.0f}, destination.options());
            result6.put_(single_idx, single_val);
            (void)result6.sum().item<float>();
        } catch (const c10::Error& e) {
            // Expected PyTorch errors are fine
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
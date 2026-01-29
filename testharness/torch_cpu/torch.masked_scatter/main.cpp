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
        
        // Create the input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the mask tensor (should be a boolean tensor with same shape as input)
        torch::Tensor mask;
        if (offset < Size) {
            mask = fuzzer_utils::createTensor(Data, Size, offset);
            // Reshape mask to match input shape if possible, or create matching shape
            try {
                mask = mask.view(input_tensor.sizes()).to(torch::kBool);
            } catch (...) {
                // If reshape fails, create mask with same shape as input
                mask = torch::ones_like(input_tensor, torch::kBool);
            }
        } else {
            // Create a mask of the same shape as input
            mask = torch::ones_like(input_tensor, torch::kBool);
        }
        
        // Count true values in mask to determine required source size
        int64_t num_true = mask.sum().item<int64_t>();
        
        // Create the source tensor with enough elements
        torch::Tensor source;
        if (offset < Size) {
            source = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert source to same dtype as input
            source = source.to(input_tensor.dtype());
            // Ensure source has enough elements
            if (source.numel() < num_true && num_true > 0) {
                // Repeat source to have enough elements
                int64_t repeat_count = (num_true / std::max(source.numel(), int64_t(1))) + 1;
                source = source.flatten().repeat({repeat_count});
            }
        } else {
            // Create a source tensor with enough values
            source = torch::ones({std::max(num_true, int64_t(1))}, input_tensor.options());
        }
        
        // Apply masked_scatter operation
        torch::Tensor result = input_tensor.masked_scatter(mask, source);
        
        // Try different variants of the operation
        if (Size > offset + 1) {
            // Create a smaller mask to test broadcasting
            if (input_tensor.dim() > 0) {
                std::vector<int64_t> smaller_shape;
                for (int64_t i = 0; i < input_tensor.dim(); i++) {
                    if (i < input_tensor.dim() - 1) {
                        smaller_shape.push_back(input_tensor.size(i));
                    } else {
                        smaller_shape.push_back(1);
                    }
                }
                
                if (!smaller_shape.empty()) {
                    try {
                        torch::Tensor smaller_mask = torch::ones(smaller_shape, torch::kBool);
                        // Need source with enough elements for broadcast mask
                        int64_t expanded_true = smaller_mask.expand(input_tensor.sizes()).sum().item<int64_t>();
                        torch::Tensor expanded_source = torch::ones({std::max(expanded_true, int64_t(1))}, input_tensor.options());
                        torch::Tensor result2 = input_tensor.masked_scatter(smaller_mask, expanded_source);
                    } catch (...) {
                        // Broadcasting may fail for certain shapes
                    }
                }
            }
            
            // Try with a scalar mask (broadcasts to all elements)
            try {
                bool scalar_mask_value = (Data[offset % Size] % 2 == 0);
                torch::Tensor scalar_mask = torch::tensor(scalar_mask_value, torch::kBool);
                int64_t needed_elements = scalar_mask_value ? input_tensor.numel() : 0;
                torch::Tensor scalar_source = torch::ones({std::max(needed_elements, int64_t(1))}, input_tensor.options());
                torch::Tensor result3 = input_tensor.masked_scatter(scalar_mask, scalar_source);
            } catch (...) {
                // Scalar mask broadcasting may have edge cases
            }
        }
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.masked_scatter_(mask, source);
        } catch (...) {
            // In-place may fail for certain configurations
        }
        
        // Try with different dtypes
        if (offset < Size) {
            try {
                torch::Tensor float_input = input_tensor.to(torch::kFloat32);
                torch::Tensor float_source = torch::ones({std::max(num_true, int64_t(1))}, torch::kFloat32);
                torch::Tensor result6 = float_input.masked_scatter(mask, float_source);
            } catch (...) {
                // Dtype conversion may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
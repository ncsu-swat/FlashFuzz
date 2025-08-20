#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor target;
        
        // If we have more data, create a target tensor with the same shape as input
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, clone the input tensor for target
            target = input.clone();
        }
        
        // Create L1Loss with different reduction modes
        uint8_t reduction_selector = 0;
        if (offset < Size) {
            reduction_selector = Data[offset++];
        }
        
        torch::nn::functional::L1LossFuncOptions::reduction_t reduction_mode;
        switch (reduction_selector % 3) {
            case 0:
                reduction_mode = torch::kNone;
                break;
            case 1:
                reduction_mode = torch::kMean;
                break;
            case 2:
                reduction_mode = torch::kSum;
                break;
        }
        
        // Create L1Loss module
        torch::nn::L1LossOptions options = torch::nn::L1LossOptions().reduction(reduction_mode);
        torch::nn::L1Loss l1_loss(options);
        
        // Apply L1Loss
        torch::Tensor loss = l1_loss(input, target);
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Set requires_grad for input
            auto input_requires_grad = input.clone().set_requires_grad(true);
            auto target_requires_grad = target.clone().set_requires_grad(true);
            
            // Recompute loss with tensors that require grad
            auto new_loss = l1_loss(input_requires_grad, target_requires_grad);
            
            // Perform backward pass
            new_loss.backward();
        }
        
        // Test with empty tensors
        if (offset < Size && Data[offset++] % 5 == 0) {
            auto empty_input = torch::empty({0});
            auto empty_target = torch::empty({0});
            try {
                auto empty_loss = l1_loss(empty_input, empty_target);
            } catch (const std::exception& e) {
                // Expected exception for empty tensors
            }
        }
        
        // Test with mismatched shapes
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                auto shape1 = std::vector<int64_t>{2, 3};
                auto shape2 = std::vector<int64_t>{3, 2};
                auto mismatched_input = torch::ones(shape1);
                auto mismatched_target = torch::ones(shape2);
                auto mismatched_loss = l1_loss(mismatched_input, mismatched_target);
            } catch (const std::exception& e) {
                // Expected exception for mismatched shapes
            }
        }
        
        // Test with different dtypes
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                auto float_input = torch::ones({2, 2}, torch::kFloat);
                auto int_target = torch::ones({2, 2}, torch::kInt);
                auto dtype_loss = l1_loss(float_input, int_target);
            } catch (const std::exception& e) {
                // May throw for incompatible dtypes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
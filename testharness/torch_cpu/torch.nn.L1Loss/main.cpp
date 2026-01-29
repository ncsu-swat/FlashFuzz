#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Track which reduction mode was selected
        int reduction_choice = reduction_selector % 3;
        
        torch::nn::L1LossOptions options;
        switch (reduction_choice) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
                options.reduction(torch::kSum);
                break;
        }
        
        // Create L1Loss module
        torch::nn::L1Loss l1_loss(options);
        
        // Apply L1Loss
        torch::Tensor loss = l1_loss(input, target);
        
        // Try backward pass if tensors require grad
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Create tensors with requires_grad
            auto input_requires_grad = input.clone().to(torch::kFloat).requires_grad_(true);
            auto target_no_grad = target.clone().to(torch::kFloat);
            
            // Recompute loss with tensors that require grad
            auto new_loss = l1_loss(input_requires_grad, target_no_grad);
            
            // For backward pass, we need a scalar. If reduction is None, sum the loss first.
            if (reduction_choice == 0) {
                new_loss = new_loss.sum();
            }
            
            // Perform backward pass
            new_loss.backward();
        }
        
        // Test with empty tensors
        if (offset < Size && Data[offset++] % 5 == 0) {
            auto empty_input = torch::empty({0});
            auto empty_target = torch::empty({0});
            try {
                auto empty_loss = l1_loss(empty_input, empty_target);
            } catch (...) {
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
            } catch (...) {
                // Expected exception for mismatched shapes
            }
        }
        
        // Test with different dtypes
        if (offset < Size && Data[offset++] % 5 == 0) {
            try {
                auto float_input = torch::ones({2, 2}, torch::kFloat);
                auto int_target = torch::ones({2, 2}, torch::kInt);
                auto dtype_loss = l1_loss(float_input, int_target);
            } catch (...) {
                // May throw for incompatible dtypes
            }
        }
        
        // Test functional API as well
        if (offset < Size && Data[offset++] % 3 == 0) {
            torch::nn::functional::L1LossFuncOptions func_options;
            switch (reduction_choice) {
                case 0:
                    func_options.reduction(torch::kNone);
                    break;
                case 1:
                    func_options.reduction(torch::kMean);
                    break;
                case 2:
                    func_options.reduction(torch::kSum);
                    break;
            }
            auto func_loss = torch::nn::functional::l1_loss(input, target, func_options);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
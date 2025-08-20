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
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, clone and modify the first one
            target = input.clone();
            // Apply some transformation to make it different
            if (target.numel() > 0) {
                target = target + 1.0;
            }
        }
        
        // Create L1Loss module with different reduction modes
        uint8_t reduction_selector = 0;
        if (offset < Size) {
            reduction_selector = Data[offset++];
        }
        
        torch::enumtype::Reduction reduction;
        switch (reduction_selector % 3) {
            case 0:
                reduction = torch::kNone;
                break;
            case 1:
                reduction = torch::kMean;
                break;
            case 2:
                reduction = torch::kSum;
                break;
        }
        
        // Create L1Loss module
        torch::nn::L1Loss l1_loss(torch::nn::L1LossOptions().reduction(reduction));
        
        // Apply the loss function
        torch::Tensor loss = l1_loss(input, target);
        
        // Try with different options
        torch::nn::L1Loss l1_loss_none(torch::nn::L1LossOptions().reduction(torch::kNone));
        torch::Tensor loss_none = l1_loss_none(input, target);
        
        torch::nn::L1Loss l1_loss_mean(torch::nn::L1LossOptions().reduction(torch::kMean));
        torch::Tensor loss_mean = l1_loss_mean(input, target);
        
        torch::nn::L1Loss l1_loss_sum(torch::nn::L1LossOptions().reduction(torch::kSum));
        torch::Tensor loss_sum = l1_loss_sum(input, target);
        
        // Test with empty tensors
        if (offset + 2 < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_input = torch::empty(empty_shape, input.options());
                torch::Tensor empty_target = torch::empty(empty_shape, target.options());
                
                torch::Tensor empty_loss = l1_loss(empty_input, empty_target);
            } catch (const std::exception& e) {
                // Just catch and continue
            }
        }
        
        // Test with mismatched shapes
        if (offset + 2 < Size) {
            try {
                std::vector<int64_t> shape1 = {2, 3};
                std::vector<int64_t> shape2 = {3, 2};
                
                torch::Tensor input1 = torch::ones(shape1, input.options());
                torch::Tensor target1 = torch::ones(shape2, target.options());
                
                torch::Tensor mismatch_loss = l1_loss(input1, target1);
            } catch (const std::exception& e) {
                // Just catch and continue
            }
        }
        
        // Test with extreme values
        if (offset + 2 < Size) {
            try {
                torch::Tensor extreme_input = torch::full_like(input, std::numeric_limits<float>::max());
                torch::Tensor extreme_target = torch::full_like(target, std::numeric_limits<float>::lowest());
                
                torch::Tensor extreme_loss = l1_loss(extreme_input, extreme_target);
            } catch (const std::exception& e) {
                // Just catch and continue
            }
        }
        
        // Test with NaN and Inf values
        if (offset + 2 < Size) {
            try {
                torch::Tensor nan_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor inf_target = torch::full_like(target, std::numeric_limits<float>::infinity());
                
                torch::Tensor special_loss = l1_loss(nan_input, inf_target);
            } catch (const std::exception& e) {
                // Just catch and continue
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
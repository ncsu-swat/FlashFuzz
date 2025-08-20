#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the target tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get reduction mode from the input data
        torch::nn::SmoothL1LossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
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
        }
        
        // Get beta value from the input data
        double beta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure beta is positive (as required by SmoothL1Loss)
            beta = std::abs(beta);
            
            // Avoid extremely large values that might cause numerical issues
            if (beta > 1e10) {
                beta = 1e10;
            }
            
            // Avoid extremely small values that might cause numerical issues
            if (beta < 1e-10) {
                beta = 1e-10;
            }
        }
        
        // Create SmoothL1Loss module with the extracted options
        torch::nn::SmoothL1LossOptions options;
        options.reduction(reduction_mode).beta(beta);
        torch::nn::SmoothL1Loss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor loss = loss_fn(input, target);
        
        // Ensure computation is completed
        loss.item<float>();
        
        // Test with different options
        if (offset < Size) {
            // Try with different beta
            double beta2 = 0.5;
            torch::nn::SmoothL1LossOptions options2;
            options2.reduction(reduction_mode).beta(beta2);
            torch::nn::SmoothL1Loss loss_fn2(options2);
            torch::Tensor loss2 = loss_fn2(input, target);
            loss2.item<float>();
            
            // Try with different reduction
            torch::nn::SmoothL1LossOptions options3;
            options3.reduction(torch::kSum).beta(beta);
            torch::nn::SmoothL1Loss loss_fn3(options3);
            torch::Tensor loss3 = loss_fn3(input, target);
            loss3.item<float>();
        }
        
        // Test the functional version
        torch::Tensor functional_loss = torch::nn::functional::smooth_l1_loss(
            input, target, torch::nn::functional::SmoothL1LossFuncOptions()
                .reduction(reduction_mode)
                .beta(beta));
        
        functional_loss.item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
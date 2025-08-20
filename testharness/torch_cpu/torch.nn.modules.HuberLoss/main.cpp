#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract delta parameter from the input data
        double delta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&delta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure delta is positive
            delta = std::abs(delta);
            if (delta == 0.0) {
                delta = 1.0;
            }
        }
        
        // Extract reduction parameter from the input data
        uint8_t reduction_byte = 0;
        if (offset < Size) {
            reduction_byte = Data[offset++];
        }
        
        // Map reduction_byte to one of the three reduction modes
        torch::nn::HuberLossOptions::reduction_t reduction;
        switch (reduction_byte % 3) {
            case 0:
                reduction = torch::kNone;
                break;
            case 1:
                reduction = torch::kMean;
                break;
            case 2:
                reduction = torch::kSum;
                break;
            default:
                reduction = torch::kMean;
        }
        
        // Create HuberLoss module with the extracted parameters
        torch::nn::HuberLossOptions options;
        options.delta(delta);
        options.reduction(reduction);
        torch::nn::HuberLoss huber_loss(options);
        
        // Apply the HuberLoss operation
        torch::Tensor loss = huber_loss->forward(input, target);
        
        // Ensure the loss is valid
        if (loss.defined() && !loss.isnan().any().item<bool>() && !loss.isinf().any().item<bool>()) {
            // Optionally perform additional operations with the loss
            auto loss_item = loss.item<double>();
        }
        
        // Try with different delta values
        if (offset + sizeof(double) <= Size) {
            double new_delta;
            std::memcpy(&new_delta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Try with a very small delta
            torch::nn::HuberLossOptions small_options;
            small_options.delta(std::abs(new_delta) * 1e-5 + 1e-10);
            small_options.reduction(reduction);
            torch::nn::HuberLoss small_huber_loss(small_options);
            torch::Tensor small_loss = small_huber_loss->forward(input, target);
            
            // Try with a very large delta
            torch::nn::HuberLossOptions large_options;
            large_options.delta(std::abs(new_delta) * 1e5 + 1.0);
            large_options.reduction(reduction);
            torch::nn::HuberLoss large_huber_loss(large_options);
            torch::Tensor large_loss = large_huber_loss->forward(input, target);
        }
        
        // Try with different reduction modes
        if (offset < Size) {
            uint8_t new_reduction_byte = Data[offset++];
            
            torch::nn::HuberLossOptions none_options;
            none_options.delta(delta);
            none_options.reduction(torch::kNone);
            torch::nn::HuberLoss none_huber_loss(none_options);
            torch::Tensor none_loss = none_huber_loss->forward(input, target);
            
            torch::nn::HuberLossOptions sum_options;
            sum_options.delta(delta);
            sum_options.reduction(torch::kSum);
            torch::nn::HuberLoss sum_huber_loss(sum_options);
            torch::Tensor sum_loss = sum_huber_loss->forward(input, target);
            
            torch::nn::HuberLossOptions mean_options;
            mean_options.delta(delta);
            mean_options.reduction(torch::kMean);
            torch::nn::HuberLoss mean_huber_loss(mean_options);
            torch::Tensor mean_loss = mean_huber_loss->forward(input, target);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
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
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (probabilities)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with the same shape as input
            target = torch::ones_like(input);
        }
        
        // Parse reduction mode from the next byte if available
        torch::enumtype::reduction_t reduction = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kMean; break;
                case 2: reduction = torch::kSum; break;
            }
        }
        
        // Parse log_target flag from the next byte if available
        bool log_target = false;
        if (offset < Size) {
            log_target = (Data[offset++] % 2) == 1;
        }
        
        // Create KLDivLoss module with different configurations
        torch::nn::KLDivLoss kl_div_loss(torch::nn::KLDivLossOptions().reduction(reduction).log_target(log_target));
        
        // Apply KLDivLoss
        torch::Tensor loss = kl_div_loss->forward(input, target);
        
        // Try with different reduction modes
        if (offset < Size) {
            uint8_t alt_reduction_byte = Data[offset++];
            torch::enumtype::reduction_t alt_reduction;
            switch (alt_reduction_byte % 3) {
                case 0: alt_reduction = torch::kNone; break;
                case 1: alt_reduction = torch::kMean; break;
                case 2: alt_reduction = torch::kSum; break;
            }
            
            if (alt_reduction != reduction) {
                torch::nn::KLDivLoss alt_kl_div_loss(torch::nn::KLDivLossOptions().reduction(alt_reduction).log_target(log_target));
                torch::Tensor alt_loss = alt_kl_div_loss->forward(input, target);
            }
        }
        
        // Try with different log_target setting
        if (offset < Size) {
            bool alt_log_target = !log_target;
            torch::nn::KLDivLoss alt_kl_div_loss(torch::nn::KLDivLossOptions().reduction(reduction).log_target(alt_log_target));
            torch::Tensor alt_loss = alt_kl_div_loss->forward(input, target);
        }
        
        // Try functional version
        int64_t reduction_int = static_cast<int64_t>(reduction);
        torch::Tensor functional_loss = torch::kl_div(input, target, reduction_int, log_target);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
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
        
        // Parse reduction mode from input data
        torch::nn::HuberLossOptions::reduction_t reduction_mode = torch::kMean;
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
        
        // Parse delta value from input data
        double delta = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&delta, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure delta is positive (required by HuberLoss)
            if (delta <= 0) {
                delta = std::abs(delta);
                if (delta == 0) {
                    delta = 1.0;
                }
            }
        }
        
        // Create HuberLoss module with options
        torch::nn::HuberLossOptions options;
        options.reduction(reduction_mode).delta(delta);
        torch::nn::HuberLoss huber_loss(options);
        
        // Apply HuberLoss
        torch::Tensor loss = huber_loss(input, target);
        
        // Ensure computation is performed by accessing the result
        float loss_value = 0.0;
        if (loss.numel() > 0) {
            if (loss.dim() == 0) {
                loss_value = loss.item<float>();
            } else {
                loss_value = loss.sum().item<float>();
            }
        }
        
        // Test with different delta values
        if (offset < Size) {
            double delta2 = 0.5;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&delta2, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                // Ensure delta is positive
                if (delta2 <= 0) {
                    delta2 = std::abs(delta2);
                    if (delta2 == 0) {
                        delta2 = 0.5;
                    }
                }
            }
            
            torch::nn::HuberLossOptions options2;
            options2.reduction(reduction_mode).delta(delta2);
            torch::nn::HuberLoss huber_loss2(options2);
            
            torch::Tensor loss2 = huber_loss2(input, target);
            
            // Access the result
            if (loss2.numel() > 0) {
                if (loss2.dim() == 0) {
                    loss_value = loss2.item<float>();
                } else {
                    loss_value = loss2.sum().item<float>();
                }
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
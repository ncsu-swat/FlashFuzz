#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (should be same shape as input)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a target with the same shape as input
            target = torch::zeros_like(input);
        }
        
        // Ensure target contains only 0s and 1s as required by MultiLabelSoftMarginLoss
        target = target.clamp(0, 1);
        
        // Parse reduction mode from the input data
        torch::nn::MultiLabelSoftMarginLossOptions options;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            int reduction_mode = reduction_byte % 3;
            
            switch (reduction_mode) {
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
        }
        
        // Parse weight tensor if we have more data
        if (offset < Size) {
            bool use_weight = Data[offset++] % 2;
            if (use_weight) {
                torch::Tensor weight;
                if (offset < Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure weight is 1D and has the right size (number of classes)
                    if (input.dim() > 1 && weight.dim() == 1) {
                        int64_t num_classes = input.size(1);
                        if (weight.size(0) == num_classes) {
                            options.weight(weight);
                        }
                    }
                }
            }
        }
        
        // Create the loss function
        torch::nn::MultiLabelSoftMarginLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn(input, target);
        
        // Ensure the output is valid
        if (output.numel() > 0) {
            float loss_value = output.item<float>();
            (void)loss_value; // Prevent unused variable warning
        }
        
        // Test backward pass if possible
        if (input.requires_grad() && output.numel() > 0 && 
            output.scalar_type() != torch::kBool && 
            output.scalar_type() != torch::kByte && 
            output.scalar_type() != torch::kInt8 && 
            output.scalar_type() != torch::kInt16 && 
            output.scalar_type() != torch::kInt32 && 
            output.scalar_type() != torch::kInt64) {
            output.backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

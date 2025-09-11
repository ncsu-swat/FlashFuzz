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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for target, create a compatible one
            if (input.dim() >= 1) {
                auto shape = input.sizes().vec();
                shape[shape.size() - 1] = 1; // Last dimension should be 1 for target
                target = torch::zeros(shape, torch::kLong);
            } else {
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Extract reduction mode from the data
        torch::nn::MultiLabelMarginLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++];
            switch (reduction_selector % 3) {
                case 0: reduction_mode = torch::kNone; break;
                case 1: reduction_mode = torch::kMean; break;
                case 2: reduction_mode = torch::kSum; break;
            }
        }
        
        // Create MultiLabelMarginLoss module
        auto options = torch::nn::MultiLabelMarginLossOptions().reduction(reduction_mode);
        auto loss_fn = torch::nn::MultiLabelMarginLoss(options);
        
        // Convert input to float if needed
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble && 
            input.scalar_type() != torch::kHalf) {
            input = input.to(torch::kFloat);
        }
        
        // Convert target to long if needed
        if (target.scalar_type() != torch::kLong) {
            target = target.to(torch::kLong);
        }
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input, target);
        
        // Try backward pass if output is a scalar or has elements
        if (output.numel() > 0) {
            try {
                output.backward();
            } catch (const std::exception&) {
                // Ignore backward exceptions
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

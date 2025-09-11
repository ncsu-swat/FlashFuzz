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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for NLLLoss2d
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        if (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Create target tensor (class indices)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has proper dimensions (N, H, W) for NLLLoss2d
            // Target should have one less dimension than input
            while (target.dim() >= input.dim()) {
                target = target.squeeze(0);
            }
            
            // Ensure target has at least 2 dimensions
            while (target.dim() < 2) {
                target = target.unsqueeze(0);
            }
            
            // Convert target to long type as required by NLLLoss2d
            target = target.to(torch::kLong);
        } else {
            // Create a default target if we don't have enough data
            auto input_sizes = input.sizes().vec();
            std::vector<int64_t> target_sizes;
            
            // Target should have dimensions [N, H, W] if input is [N, C, H, W]
            if (input.dim() >= 3) {
                target_sizes.push_back(input_sizes[0]); // N
                for (size_t i = 2; i < input_sizes.size(); i++) {
                    target_sizes.push_back(input_sizes[i]); // H, W, ...
                }
            } else {
                target_sizes = {1, 1}; // Fallback
            }
            
            target = torch::zeros(target_sizes, torch::kLong);
        }
        
        // Get weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (offset < Size) {
                use_weight = true;
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weight is 1D and has size equal to number of classes
                if (input.dim() >= 2) {
                    int64_t num_classes = input.size(1);
                    weight = weight.flatten();
                    if (weight.size(0) != num_classes) {
                        weight = weight.repeat(num_classes / (weight.size(0) + 1) + 1);
                        weight = weight.slice(0, 0, num_classes);
                    }
                }
            }
        }
        
        // Parse reduction mode
        torch::nn::NLLLossOptions::reduction_t reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kSum;
                    break;
                case 2:
                default:
                    reduction_mode = torch::kMean;
                    break;
            }
        }
        
        // Parse ignore_index
        int64_t ignore_index = -100; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&ignore_index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create NLLLoss2d options
        auto options = torch::nn::NLLLossOptions()
            .reduction(reduction_mode)
            .ignore_index(ignore_index);
        
        if (use_weight) {
            options = options.weight(weight);
        }
        
        // Create NLLLoss2d module
        torch::nn::NLLLoss nll_loss2d(options);
        
        // Apply NLLLoss2d
        torch::Tensor output = nll_loss2d(input, target);
        
        // Ensure we use the output to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

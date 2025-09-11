#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (log probabilities)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for NLLLoss2d
        if (input.dim() < 4) {
            input = input.reshape({1, 2, 3, 3});
        }
        
        // Create target tensor (class indices)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has correct dimensions (N, H, W) for NLLLoss2d
            if (target.dim() != input.dim() - 1) {
                std::vector<int64_t> target_shape;
                for (int i = 0; i < input.dim(); i++) {
                    if (i != 1) { // Skip the C dimension
                        target_shape.push_back(input.size(i));
                    }
                }
                target = target.reshape(target_shape);
            }
            
            // Convert target to Long type and ensure valid class indices
            target = target.to(torch::kLong).abs() % input.size(1);
        } else {
            // Create a default target if we don't have enough data
            std::vector<int64_t> target_shape;
            for (int i = 0; i < input.dim(); i++) {
                if (i != 1) { // Skip the C dimension
                    target_shape.push_back(input.size(i));
                }
            }
            target = torch::zeros(target_shape, torch::kLong);
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weight has correct dimension (C) for NLLLoss2d
                if (input.dim() >= 2) {
                    weight = weight.reshape({input.size(1)});
                    use_weight = true;
                }
            }
        }
        
        // Parse reduction mode
        int64_t reduction_mode = 1; // Mean
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = 0; // None
                    break;
                case 1:
                    reduction_mode = 1; // Mean
                    break;
                case 2:
                    reduction_mode = 2; // Sum
                    break;
            }
        }
        
        // Parse ignore_index
        int64_t ignore_index = -100; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&ignore_index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply the NLLLoss2d operation using functional interface
        torch::Tensor output;
        if (use_weight) {
            output = torch::nll_loss2d(input, target, weight, reduction_mode, ignore_index);
        } else {
            output = torch::nll_loss2d(input, target, {}, reduction_mode, ignore_index);
        }
        
        // Ensure we've consumed the output to prevent optimization
        volatile float sum = output.sum().item<float>();
        (void)sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

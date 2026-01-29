#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least a few bytes to create meaningful tensors
        if (Size < 8) {
            return 0;
        }
        
        // Parse dimensions from input
        int64_t N = 1 + (Data[offset++] % 4);  // Batch size: 1-4
        int64_t C = 2 + (Data[offset++] % 8);  // Classes: 2-9
        int64_t H = 2 + (Data[offset++] % 8);  // Height: 2-9
        int64_t W = 2 + (Data[offset++] % 8);  // Width: 2-9
        
        // Create input tensor (N, C, H, W) - should be log probabilities
        torch::Tensor input = torch::randn({N, C, H, W}, torch::kFloat32);
        
        // Apply log_softmax along the class dimension to get log probabilities
        input = torch::log_softmax(input, /*dim=*/1);
        
        // Create target tensor (N, H, W) with valid class indices [0, C)
        torch::Tensor target = torch::randint(0, C, {N, H, W}, torch::kLong);
        
        // Optionally modify target based on fuzzer data
        if (offset < Size) {
            torch::Tensor target_mod = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Try to use fuzzer-generated values for target
                target_mod = target_mod.to(torch::kLong).abs() % C;
                if (target_mod.numel() >= N * H * W) {
                    target = target_mod.flatten().slice(0, 0, N * H * W).reshape({N, H, W});
                }
            } catch (...) {
                // Keep the default target on reshape failure
            }
        }
        
        // Create weight tensor (optional)
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Weight should be of size C (number of classes)
            weight = torch::rand({C}, torch::kFloat32) + 0.1f;  // Avoid zero weights
            use_weight = true;
        }
        
        // Parse reduction mode
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction = torch::Reduction::None;
                    break;
                case 1:
                    reduction = torch::Reduction::Mean;
                    break;
                case 2:
                    reduction = torch::Reduction::Sum;
                    break;
            }
        }
        
        // Parse ignore_index
        int64_t ignore_index = -100; // Default value
        if (offset < Size) {
            // Sometimes use a valid class index to ignore
            if (Data[offset] % 4 == 0) {
                ignore_index = Data[offset] % C;
            }
            offset++;
        }
        
        // Apply the NLLLoss2d operation
        torch::Tensor output;
        if (use_weight) {
            output = torch::nll_loss2d(input, target, weight, reduction, ignore_index);
        } else {
            output = torch::nll_loss2d(input, target, {}, reduction, ignore_index);
        }
        
        // Ensure we've consumed the output to prevent optimization
        if (reduction == torch::Reduction::None) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        } else {
            volatile float val = output.item<float>();
            (void)val;
        }
        
        // Also test with different input values derived from fuzzer data
        if (offset < Size) {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
            try {
                // Reshape and normalize fuzzer input
                int64_t numel_needed = N * C * H * W;
                if (fuzz_input.numel() >= numel_needed) {
                    fuzz_input = fuzz_input.flatten().slice(0, 0, numel_needed).reshape({N, C, H, W}).to(torch::kFloat32);
                    fuzz_input = torch::log_softmax(fuzz_input, 1);
                    
                    torch::Tensor output2;
                    if (use_weight) {
                        output2 = torch::nll_loss2d(fuzz_input, target, weight, reduction, ignore_index);
                    } else {
                        output2 = torch::nll_loss2d(fuzz_input, target, {}, reduction, ignore_index);
                    }
                    
                    if (reduction == torch::Reduction::None) {
                        volatile float sum2 = output2.sum().item<float>();
                        (void)sum2;
                    } else {
                        volatile float val2 = output2.item<float>();
                        (void)val2;
                    }
                }
            } catch (...) {
                // Silently ignore shape mismatch errors for fuzzer-derived inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
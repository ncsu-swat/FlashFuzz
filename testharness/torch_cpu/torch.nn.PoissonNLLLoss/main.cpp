#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Extract configuration parameters first
        bool log_input = Data[offset++] & 0x1;
        bool full = Data[offset++] & 0x1;
        uint8_t reduction_selector = Data[offset++] % 3;
        bool do_backward = Data[offset++] & 0x1;
        
        // Extract eps value
        float eps_raw;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
        } else {
            eps_raw = 1e-8f;
        }
        double eps = std::abs(static_cast<double>(eps_raw));
        if (eps < 1e-12 || std::isnan(eps) || std::isinf(eps)) {
            eps = 1e-8;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input (required for loss computation)
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input and target have matching shapes
        try {
            target = target.view(input.sizes());
        } catch (...) {
            // If reshape fails, create target with same shape
            target = torch::rand_like(input);
        }
        
        // Target for Poisson should be non-negative (counts)
        target = torch::abs(target);
        
        // If log_input is false, input represents the rate directly and should be positive
        if (!log_input) {
            input = torch::abs(input) + eps;
        }
        
        // Enable gradients for backward pass testing
        input = input.detach().requires_grad_(true);
        
        // Create PoissonNLLLoss module with options based on reduction selector
        torch::nn::PoissonNLLLossOptions options;
        options.log_input(log_input);
        options.full(full);
        options.eps(eps);
        
        // Set reduction using the correct variant types
        switch (reduction_selector) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
            default:
                options.reduction(torch::kSum);
                break;
        }
        
        auto poisson_loss = torch::nn::PoissonNLLLoss(options);
        
        // Apply the loss function
        torch::Tensor loss = poisson_loss->forward(input, target);
        
        // Verify loss is computed
        if (loss.defined()) {
            // Force computation
            float loss_val = loss.sum().item<float>();
            (void)loss_val;
            
            // Optionally perform a backward pass to test gradients
            if (do_backward && input.requires_grad()) {
                try {
                    loss.sum().backward();
                    // Access gradient to ensure it's computed
                    if (input.grad().defined()) {
                        float grad_sum = input.grad().sum().item<float>();
                        (void)grad_sum;
                    }
                } catch (...) {
                    // Backward pass may fail for certain configurations
                }
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
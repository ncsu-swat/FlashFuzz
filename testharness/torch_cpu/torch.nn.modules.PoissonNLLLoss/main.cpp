#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        // Extract configuration parameters from the remaining data
        bool full = false;
        bool log_input = false;
        double eps = 1e-8;
        int reduction_type = 1; // Default to Mean
        
        if (offset < Size) {
            full = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            log_input = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            // Extract eps value, ensure it's positive
            memcpy(&eps, Data + offset, sizeof(double));
            eps = std::abs(eps);
            if (eps == 0.0 || std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-8; // Avoid invalid values
            }
            // Clamp eps to reasonable range
            eps = std::max(1e-12, std::min(eps, 1.0));
            offset += sizeof(double);
        }
        
        if (offset < Size) {
            reduction_type = Data[offset++] % 3;
        }
        
        // Make sure input and target have compatible shapes
        if (input.dim() > 0 && target.dim() > 0) {
            // Try to make shapes compatible for the loss function
            if (input.sizes() != target.sizes()) {
                // Reshape target to match input if possible
                if (input.numel() == target.numel()) {
                    target = target.reshape_as(input);
                } else {
                    // If different number of elements, create a new target with same shape as input
                    target = torch::ones_like(input);
                }
            }
        }
        
        // Ensure inputs are non-negative for Poisson distribution
        // This is a requirement for the mathematical model
        input = torch::abs(input) + eps; // Add eps to avoid log(0)
        target = torch::abs(target);
        
        // Create the PoissonNLLLoss module with appropriate reduction
        torch::nn::PoissonNLLLossOptions options;
        options.log_input(log_input);
        options.full(full);
        options.eps(eps);
        
        switch (reduction_type) {
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
        
        torch::nn::PoissonNLLLoss poisson_loss(options);
        
        // Apply the loss function
        torch::Tensor loss = poisson_loss->forward(input, target);
        
        // Perform some operations on the result to ensure it's used
        if (loss.defined()) {
            // Use sum() to handle both scalar and non-scalar cases (when reduction=None)
            auto total_loss = loss.sum().item<float>();
            (void)total_loss; // Suppress unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
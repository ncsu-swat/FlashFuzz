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
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract configuration parameters from the remaining data
        bool full = false;
        bool log_input = false;
        double eps = 1e-8;
        torch::enumtype::reduction_t reduction = torch::kMean;
        
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
            if (eps == 0.0) eps = 1e-8; // Avoid division by zero
            offset += sizeof(double);
        }
        
        if (offset < Size) {
            uint8_t red_val = Data[offset++] % 3;
            switch (red_val) {
                case 0: reduction = torch::kNone; break;
                case 1: reduction = torch::kSum; break;
                default: reduction = torch::kMean; break;
            }
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
        // This is not a sanity check but a requirement for the mathematical model
        input = torch::abs(input);
        target = torch::abs(target);
        
        // Create the PoissonNLLLoss module
        torch::nn::PoissonNLLLoss poisson_loss(
            torch::nn::PoissonNLLLossOptions()
                .log_input(log_input)
                .full(full)
                .eps(eps)
                .reduction(reduction)
        );
        
        // Apply the loss function
        torch::Tensor loss = poisson_loss->forward(input, target);
        
        // Perform some operations on the result to ensure it's used
        if (loss.defined()) {
            auto item_loss = loss.item<float>();
            if (std::isnan(item_loss) || std::isinf(item_loss)) {
                // Just noting the NaN/Inf, but not throwing an exception
                // This allows the fuzzer to explore these cases
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

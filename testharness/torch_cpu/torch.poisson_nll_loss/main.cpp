#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

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
        
        // Need at least some data to create tensors and options
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor - reuse remaining data
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract boolean options from remaining data
        bool full = false;
        bool log_input = false;
        
        if (offset < Size) {
            full = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            log_input = Data[offset++] & 0x1;
        }
        
        // Extract epsilon value (small positive number)
        double eps = 1e-8;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            std::memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and reasonable
            eps = static_cast<double>(std::abs(eps_f));
            if (eps < 1e-12 || std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-8;
            }
            if (eps > 1.0) {
                eps = 1.0;
            }
        }
        
        // Determine reduction type
        int64_t reduction = 1; // Mean by default (0=None, 1=Mean, 2=Sum)
        if (offset < Size) {
            reduction = Data[offset++] % 3;
        }
        
        // Inner try-catch for expected operational failures
        try
        {
            // Make target match input shape for valid comparison
            if (input.sizes() != target.sizes()) {
                target = target.reshape_as(input);
            }
            
            // For Poisson NLL loss:
            // - target should be non-negative (Poisson counts)
            // - if log_input=false, input represents rates and should be positive
            // - if log_input=true, input represents log-rates
            target = torch::abs(target);
            
            if (!log_input) {
                // input is rate, must be positive
                input = torch::abs(input) + eps;
            }
            // else input is log-rate, can be any value
            
            // Apply the poisson_nll_loss operation
            torch::Tensor result = torch::poisson_nll_loss(
                input, 
                target, 
                log_input,  // log_input comes before full in PyTorch C++ API
                full, 
                eps, 
                reduction
            );
            
            // Ensure the result is computed
            if (result.numel() > 0) {
                if (reduction == 0) {
                    // No reduction - result has same shape as input
                    auto sum = result.sum().item<float>();
                    (void)sum;
                } else {
                    auto item = result.item<float>();
                    (void)item;
                }
            }
        }
        catch (const c10::Error &e)
        {
            // Silently catch expected PyTorch errors (shape mismatch, etc.)
        }
        catch (const std::runtime_error &e)
        {
            // Silently catch runtime errors from invalid operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}
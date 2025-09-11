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
        
        // Extract configuration parameters from remaining data
        bool full_reduction = true;
        bool log_input = true;
        double eps = 1e-8;
        
        if (offset + 2 < Size) {
            full_reduction = Data[offset++] & 0x1;
            log_input = Data[offset++] & 0x1;
            
            // Extract eps value (ensure it's positive)
            if (offset + sizeof(float) <= Size) {
                float eps_raw;
                std::memcpy(&eps_raw, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Ensure eps is positive and not too small
                eps = std::abs(eps_raw);
                if (eps < 1e-12) {
                    eps = 1e-8;
                }
            }
        }
        
        // Determine reduction mode
        torch::nn::PoissonNLLLossOptions::reduction_t reduction_mode;
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++] % 3;
            switch (reduction_selector) {
                case 0:
                    reduction_mode = torch::nn::PoissonNLLLossOptions::reduction_t::kNone;
                    break;
                case 1:
                    reduction_mode = torch::nn::PoissonNLLLossOptions::reduction_t::kMean;
                    break;
                case 2:
                default:
                    reduction_mode = torch::nn::PoissonNLLLossOptions::reduction_t::kSum;
                    break;
            }
        } else {
            reduction_mode = full_reduction ? torch::nn::PoissonNLLLossOptions::reduction_t::kMean : torch::nn::PoissonNLLLossOptions::reduction_t::kNone;
        }
        
        // Create PoissonNLLLoss module
        torch::nn::PoissonNLLLossOptions options;
        options.log_input(log_input)
               .full(full_reduction)
               .eps(eps)
               .reduction(reduction_mode);
        
        auto poisson_loss = torch::nn::PoissonNLLLoss(options);
        
        // Apply the loss function
        torch::Tensor loss = poisson_loss->forward(input, target);
        
        // Ensure loss is computed and not NaN
        if (loss.defined() && !loss.isnan().any().item<bool>()) {
            // Optionally perform a backward pass to test gradients
            if (input.requires_grad() && offset < Size && (Data[offset++] & 0x1)) {
                loss.backward();
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

#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input, variance, and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor variance = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure variance is positive (required by GaussianNLLLoss)
        variance = torch::abs(variance) + 1e-6;
        
        // Parse reduction mode from the remaining data
        torch::Reduction::Reduction reduction_mode = torch::kMean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            switch (reduction_byte % 3) {
                case 0:
                    reduction_mode = torch::kNone;
                    break;
                case 1:
                    reduction_mode = torch::kMean;
                    break;
                case 2:
                    reduction_mode = torch::kSum;
                    break;
            }
        }
        
        // Parse full parameter
        bool full = false;
        if (offset < Size) {
            full = Data[offset++] & 1;
        }
        
        // Parse eps parameter
        double eps = 1e-6;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive and not too large
            eps = std::abs(eps);
            if (eps < 1e-12) eps = 1e-12;
            if (eps > 1.0) eps = 1.0;
        }
        
        // Try to make input, variance, and target compatible if possible
        if (input.dim() > 0 && variance.dim() > 0 && target.dim() > 0) {
            // Convert to same dtype if needed
            if (input.scalar_type() != target.scalar_type()) {
                if (torch::isFloatingType(input.scalar_type())) {
                    target = target.to(input.scalar_type());
                } else if (torch::isFloatingType(target.scalar_type())) {
                    input = input.to(target.scalar_type());
                } else {
                    input = input.to(torch::kFloat);
                    target = target.to(torch::kFloat);
                }
            }
            
            // Convert variance to same dtype
            variance = variance.to(input.scalar_type());
        }
        
        // Apply the loss function using functional API
        torch::Tensor loss = torch::nn::functional::gaussian_nll_loss(
            input, target, variance, 
            torch::nn::functional::GaussianNLLLossFuncOptions()
                .full(full)
                .eps(eps)
                .reduction(reduction_mode)
        );
        
        // Perform backward pass if possible
        if (loss.numel() > 0 && torch::isFloatingType(loss.scalar_type())) {
            try {
                loss.backward();
            } catch (const std::exception &e) {
                // Backward pass failed, but we can continue
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

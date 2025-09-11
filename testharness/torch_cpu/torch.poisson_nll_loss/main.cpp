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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract boolean options from remaining data
        bool full = false;
        bool log_input = false;
        bool reduction_none = false;
        
        if (offset < Size) {
            full = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            log_input = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            reduction_none = Data[offset++] & 0x1;
        }
        
        // Extract epsilon value (small positive number)
        double eps = 1e-8;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure epsilon is positive and not too large
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-8;
            if (eps > 1.0) eps = 1.0;
        }
        
        // Determine reduction type
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t red_byte = Data[offset++];
            if (red_byte % 3 == 0) {
                reduction = torch::Reduction::None;
            } else if (red_byte % 3 == 1) {
                reduction = torch::Reduction::Mean;
            } else {
                reduction = torch::Reduction::Sum;
            }
        }
        
        // Make sure input and target have non-negative values for Poisson distribution
        input = torch::abs(input);
        target = torch::abs(target);
        
        // Apply the poisson_nll_loss operation
        torch::Tensor result = torch::poisson_nll_loss(
            input, 
            target, 
            full, 
            eps, 
            reduction, 
            log_input
        );
        
        // Ensure the result is valid
        if (result.numel() > 0) {
            auto item = result.item<float>();
            (void)item; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

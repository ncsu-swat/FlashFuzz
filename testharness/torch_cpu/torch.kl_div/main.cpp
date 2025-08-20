#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for kl_div
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor with same shape as input
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse reduction mode from input data
        int64_t reduction_mode = 0;
        if (offset < Size) {
            reduction_mode = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Parse log_target flag from input data
        bool log_target = false;
        if (offset < Size) {
            log_target = Data[offset++] & 0x1;
        }
        
        // Convert reduction mode to int64_t
        int64_t reduction;
        switch (reduction_mode) {
            case 0:
                reduction = at::Reduction::None;
                break;
            case 1:
                reduction = at::Reduction::Mean;
                break;
            case 2:
                reduction = at::Reduction::Sum;
                break;
            default:
                reduction = at::Reduction::None;
        }
        
        // Apply kl_div operation
        torch::Tensor result = torch::kl_div(input, target, reduction, log_target);
        
        // Try different variants of the function
        if (offset < Size) {
            // Try with default parameters
            torch::Tensor result2 = torch::kl_div(input, target);
            
            // Try with only reduction specified
            torch::Tensor result3 = torch::kl_div(input, target, reduction);
        }
        
        // Try functional variant
        torch::nn::functional::KLDivFuncOptions options;
        if (reduction == at::Reduction::None) {
            options.reduction(torch::kNone);
        } else if (reduction == at::Reduction::Mean) {
            options.reduction(torch::kMean);
        } else if (reduction == at::Reduction::Sum) {
            options.reduction(torch::kSum);
        }
        options.log_target(log_target);
        torch::Tensor result4 = torch::nn::functional::kl_div(input, target, options);
        
        // Try with different tensor types
        if (offset + 4 < Size) {
            torch::Tensor input_float = input.to(torch::kFloat);
            torch::Tensor target_float = target.to(torch::kFloat);
            torch::Tensor result5 = torch::kl_div(input_float, target_float, reduction, log_target);
            
            // Try with double precision
            torch::Tensor input_double = input.to(torch::kDouble);
            torch::Tensor target_double = target.to(torch::kDouble);
            torch::Tensor result6 = torch::kl_div(input_double, target_double, reduction, log_target);
            
            // Try with half precision if supported
            if (torch::cuda::is_available()) {
                torch::Tensor input_half = input.to(torch::kHalf);
                torch::Tensor target_half = target.to(torch::kHalf);
                torch::Tensor result7 = torch::kl_div(input_half, target_half, reduction, log_target);
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
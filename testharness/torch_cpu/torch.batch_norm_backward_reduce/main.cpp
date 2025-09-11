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
        
        // Create input tensors for batch_norm_backward_reduce
        torch::Tensor grad_out = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input tensor for x
        torch::Tensor x;
        if (offset < Size) {
            x = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with the same shape as grad_out
            x = torch::ones_like(grad_out);
        }
        
        // Create mean tensor (typically 1D with size equal to channels)
        torch::Tensor mean;
        if (offset < Size) {
            mean = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default mean tensor
            int64_t channels = grad_out.dim() > 1 ? grad_out.size(1) : 1;
            mean = torch::zeros({channels}, grad_out.options());
        }
        
        // Create invstd tensor (typically 1D with size equal to channels)
        torch::Tensor invstd;
        if (offset < Size) {
            invstd = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default invstd tensor
            int64_t channels = grad_out.dim() > 1 ? grad_out.size(1) : 1;
            invstd = torch::ones({channels}, grad_out.options());
        }
        
        // Create weight tensor (optional)
        std::optional<torch::Tensor> weight;
        if (offset < Size && (Data[offset++] & 1)) {
            if (offset < Size) {
                weight = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                int64_t channels = grad_out.dim() > 1 ? grad_out.size(1) : 1;
                weight = torch::ones({channels}, grad_out.options());
            }
        }
        
        // Parse gradient flags
        bool input_g = true;
        bool weight_g = true;
        bool bias_g = true;
        
        if (offset < Size) {
            input_g = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            weight_g = Data[offset++] & 1;
        }
        
        if (offset < Size) {
            bias_g = Data[offset++] & 1;
        }
        
        // Call batch_norm_backward_reduce
        auto result = torch::batch_norm_backward_reduce(
            grad_out,
            x,
            mean,
            invstd,
            weight,
            input_g,
            weight_g,
            bias_g
        );
        
        // Unpack the result tuple
        auto sum_dy = std::get<0>(result);
        auto sum_dy_xmu = std::get<1>(result);
        auto grad_weight = std::get<2>(result);
        auto grad_bias = std::get<3>(result);
        
        // Perform some operations on the results to ensure they're used
        auto sum = sum_dy.sum() + sum_dy_xmu.sum();
        if (weight_g) {
            sum += grad_weight.sum();
        }
        if (bias_g) {
            sum += grad_bias.sum();
        }
        
        // Prevent compiler from optimizing away the computation
        if (std::isnan(sum.item<double>())) {
            throw std::runtime_error("NaN detected in result");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}

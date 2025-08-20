#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor (TBC format: Time, Batch, Channel)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for TBC format
        if (input.dim() < 3) {
            input = input.reshape({
                input.numel() > 0 ? 1 : 0,  // Time
                input.numel() > 0 ? 1 : 0,  // Batch
                input.numel() > 0 ? input.numel() : 0  // Channel
            });
        }
        
        // Create weight tensor (kernel)
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure weight has correct shape for conv_tbc: [kernel_width, in_channels, out_channels]
        if (weight.dim() != 3 && weight.numel() > 0) {
            int64_t total_elements = weight.numel();
            int64_t kernel_width = std::max(int64_t(1), std::min(int64_t(5), total_elements > 0 ? total_elements % 5 + 1 : 1));
            int64_t in_channels = input.dim() >= 3 ? input.size(2) : 1;
            int64_t out_channels = std::max(int64_t(1), total_elements / (kernel_width * std::max(int64_t(1), in_channels)));
            
            weight = weight.reshape({kernel_width, in_channels, out_channels});
        }
        
        // Create bias tensor
        torch::Tensor bias;
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure bias has correct shape: [out_channels]
            if (weight.dim() == 3) {
                int64_t out_channels = weight.size(2);
                if (bias.numel() != out_channels) {
                    bias = bias.reshape({out_channels});
                }
            }
        } else {
            // If we don't have enough data for bias, create a zero tensor
            if (weight.dim() == 3) {
                bias = torch::zeros({weight.size(2)}, weight.options());
            } else {
                bias = torch::zeros({1}, weight.options());
            }
        }
        
        // Get padding value
        int64_t pad = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&pad, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Limit padding to reasonable values
            pad = std::abs(pad) % 10;
        }
        
        // Try to apply conv_tbc
        torch::Tensor output = torch::conv_tbc(input, weight, bias, pad);
        
        // Use the output to prevent optimization from removing the computation
        if (output.defined() && output.numel() > 0) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
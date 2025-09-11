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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple weight tensor
            weight = torch::ones({1, input.size(1), 3, 3});
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else if (use_bias) {
            // Create a simple bias if we need one but don't have data
            bias = torch::ones({weight.size(0)});
        }
        
        // Parse stride, padding, dilation, groups
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> padding = {0, 0};
        std::vector<int64_t> dilation = {1, 1};
        int64_t groups = 1;
        
        if (offset + 4 < Size) {
            stride[0] = (Data[offset] % 3) + 1;
            stride[1] = (Data[offset + 1] % 3) + 1;
            padding[0] = Data[offset + 2] % 3;
            padding[1] = Data[offset + 3] % 3;
            offset += 4;
        }
        
        if (offset + 3 < Size) {
            dilation[0] = (Data[offset] % 2) + 1;
            dilation[1] = (Data[offset + 1] % 2) + 1;
            groups = (Data[offset + 2] % 4) + 1;
            offset += 3;
        }
        
        // Try to make input and weight compatible for convolution
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        if (weight.dim() < 4) {
            weight = weight.reshape({1, input.size(1), 3, 3});
        }
        
        // Ensure weight has compatible dimensions with input
        if (weight.size(1) != input.size(1) / groups) {
            weight = weight.reshape({weight.size(0), input.size(1) / groups, weight.size(2), weight.size(3)});
        }
        
        // Ensure bias has compatible dimensions with weight
        if (use_bias && bias.numel() != weight.size(0)) {
            bias = bias.reshape({weight.size(0)});
        }
        
        // Apply mkldnn_convolution
        torch::Tensor output;
        if (use_bias) {
            output = torch::mkldnn_convolution(
                input, weight, bias, stride, padding, dilation, groups
            );
        } else {
            output = torch::mkldnn_convolution(
                input, weight, {}, stride, padding, dilation, groups
            );
        }
        
        // Ensure the output is materialized
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
